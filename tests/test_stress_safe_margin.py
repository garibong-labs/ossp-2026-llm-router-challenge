# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Checks for the development-only composition stress tool.

The regression thresholds in :class:`StressPublicSplitTest` are the acceptance
gates for the safe-margin retune. They are deliberately stated as inequalities
against the deterministic 500-resample run so that a future change to the
planner cannot quietly reintroduce the concentration tail.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import pathlib
import sys
import tempfile
import unittest

from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy, load_input
from ossp_router.scoring import score_submissions


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOY_INPUT = ROOT / "data/toy/inputs.json"
TOY_OUTCOMES = ROOT / "data/toy/outcomes.json"
DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"
TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


stress = _load_module("stress_safe_margin", ROOT / "tools/stress_safe_margin.py")


def _toy_report(**overrides):
    arguments = {
        "input_path": TOY_INPUT,
        "outcomes_path": TOY_OUTCOMES,
        "resamples": 8,
        "seed": 11,
    }
    arguments.update(overrides)
    return stress.stress_split(**arguments)


class BootstrapDrawTest(unittest.TestCase):
    """The resampling itself must be reproducible from the recorded seed."""

    def test_the_same_seed_reproduces_the_same_draws(self):
        first = stress.bootstrap_indices(50, 7, 20260822)
        second = stress.bootstrap_indices(50, 7, 20260822)
        self.assertEqual(first, second)

    def test_a_different_seed_changes_the_draws(self):
        first = stress.bootstrap_indices(50, 7, 20260822)
        second = stress.bootstrap_indices(50, 7, 20260823)
        self.assertNotEqual(first, second)

    def test_each_draw_has_the_sample_size_and_stays_in_range(self):
        draws = stress.bootstrap_indices(12, 5, 3)
        self.assertEqual(5, len(draws))
        for indices in draws:
            self.assertEqual(12, len(indices))
            self.assertTrue(all(0 <= index < 12 for index in indices))

    def test_drawing_is_with_replacement(self):
        draws = stress.bootstrap_indices(30, 20, 20260822)
        self.assertTrue(
            any(len(set(indices)) < len(indices) for indices in draws),
            "복원 추출이라면 중복이 나타나야 합니다.",
        )

    def test_a_split_is_seeded_independently_of_any_other_split(self):
        # A shared generator would make a split's report depend on whether
        # another split happened to run first in the same process.
        before = stress.bootstrap_indices(40, 4, 20260822)
        stress.bootstrap_indices(97, 30, 20260822)
        self.assertEqual(before, stress.bootstrap_indices(40, 4, 20260822))

    def test_an_empty_or_negative_request_is_rejected(self):
        for size, resamples in ((0, 5), (5, 0), (-1, 5), (5, -1)):
            with self.subTest(size=size, resamples=resamples):
                with self.assertRaises(ValueError):
                    stress.bootstrap_indices(size, resamples, 1)


class QuantileTest(unittest.TestCase):
    """Nearest-rank quantiles, as declared in the report."""

    def test_nearest_rank_picks_the_documented_element(self):
        ordered = [float(value) for value in range(1, 101)]
        self.assertEqual(50.0, stress.quantile(ordered, 0.50))
        self.assertEqual(95.0, stress.quantile(ordered, 0.95))
        self.assertEqual(99.0, stress.quantile(ordered, 0.99))
        self.assertEqual(100.0, stress.quantile(ordered, 1.0))

    def test_the_lowest_quantile_never_falls_off_the_front(self):
        self.assertEqual(1.0, stress.quantile([1.0, 2.0, 3.0], 0.0))

    def test_an_empty_sequence_is_rejected(self):
        with self.assertRaises(ValueError):
            stress.quantile([], 0.5)


class ReportSchemaTest(unittest.TestCase):
    """The report must record everything needed to rerun the audit."""

    @classmethod
    def setUpClass(cls):
        cls.report = _toy_report()

    def test_the_report_records_the_audit_configuration(self):
        self.assertEqual(stress.REPORT_TYPE, self.report["report_type"])
        self.assertEqual(11, self.report["seed"])
        self.assertEqual(8, self.report["resamples"])
        self.assertEqual(stress.QUANTILE_METHOD, self.report["quantile_method"])
        self.assertEqual(list(stress.QUANTILES), self.report["quantiles"])
        self.assertTrue(self.report["with_replacement"])
        self.assertEqual(
            self.report["num_episodes"], self.report["sample_size"]
        )

    def test_every_tier_reports_a_distribution_and_a_breach_count(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                entry = self.report["bootstrap"][tier]
                self.assertEqual(8, entry["resamples"])
                for field in ("mean", "min", "p50", "p95", "p99", "max"):
                    self.assertIn(field, entry["cost_ratio"])
                    self.assertIn(field, entry["quality_score"])
                self.assertGreaterEqual(entry["cap_breaches"], 0)
                self.assertLessEqual(entry["cap_breaches"], 8)
                self.assertIn("worst_resample", entry)
                self.assertIn(tier, self.report["whole_split"])
                self.assertEqual(
                    sorted(MODEL_IDS),
                    sorted(self.report["whole_split"][tier]["model_counts"]),
                )

    def test_the_distribution_is_internally_ordered(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                stats = self.report["bootstrap"][tier]["cost_ratio"]
                self.assertLessEqual(stats["min"], stats["p50"])
                self.assertLessEqual(stats["p50"], stats["p95"])
                self.assertLessEqual(stats["p95"], stats["p99"])
                self.assertLessEqual(stats["p99"], stats["max"])

    def test_the_total_breach_count_matches_the_per_tier_counts(self):
        self.assertEqual(
            sum(self.report["bootstrap"][tier]["cap_breaches"] for tier in TIERS),
            self.report["total_cap_breaches"],
        )

    def test_the_report_states_that_it_is_evidence_and_not_a_guarantee(self):
        self.assertIn("보장하지 않습니다", self.report["evidence_scope"])

    def test_the_report_never_stores_a_local_absolute_path(self):
        for key in ("input_path", "outcomes_path", "artifact_path"):
            with self.subTest(key=key):
                self.assertFalse(self.report[key].startswith("/"))

    def test_the_report_carries_no_episode_identifier(self):
        episode_ids = {
            episode.episode_id for episode in load_input(TOY_INPUT).episodes
        }
        serialized = json.dumps(self.report, ensure_ascii=False)
        for episode_id in episode_ids:
            with self.subTest(episode_id=episode_id):
                self.assertNotIn(episode_id, serialized)


class ReportDeterminismTest(unittest.TestCase):
    """Two runs of the same command must produce identical bytes."""

    def test_repeated_runs_produce_the_same_report(self):
        self.assertEqual(_toy_report(), _toy_report())

    def test_the_written_report_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary)
            written = []
            for name in ("first.json", "second.json"):
                path = target / name
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    code = stress.main(
                        [
                            "--input",
                            str(TOY_INPUT),
                            "--outcomes",
                            str(TOY_OUTCOMES),
                            "--resamples",
                            "8",
                            "--seed",
                            "11",
                            "--report",
                            str(path),
                            "--quiet",
                        ]
                    )
                self.assertIn(code, (0, 1))
                written.append(path.read_bytes())
            self.assertEqual(written[0], written[1])

    def test_a_missing_input_is_reported_without_a_traceback(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = stress.main(
                [
                    "--input",
                    str(ROOT / "data/toy/does-not-exist.json"),
                    "--outcomes",
                    str(TOY_OUTCOMES),
                    "--quiet",
                ]
            )
        self.assertEqual(2, code)


class WholeSplitAgreementTest(unittest.TestCase):
    """The tool's float cost ratio must agree with the official Decimal scorer."""

    def test_the_stress_ratio_matches_the_official_scoring_path(self):
        fixture = stress.prepare_split(
            input_path=TOY_INPUT, outcomes_path=TOY_OUTCOMES
        )
        safe_margin = sys.modules["safe_margin"]
        inputs = load_input(TOY_INPUT)
        policy = load_bundled_policy()
        artifact = safe_margin.load_artifact(stress.DEFAULT_ARTIFACT)
        submissions = [
            safe_margin.make_safe_margin_submission(
                inputs, policy, artifact, tier
            ).submission
            for tier in TIERS
        ]
        official = score_submissions(
            inputs, stress.load_outcomes(TOY_OUTCOMES), submissions, policy
        )
        for tier in TIERS:
            with self.subTest(tier=tier):
                expected = float(official["tiers"][tier]["budget_ratio"])
                measured = stress.whole_split_tier(fixture, tier)["cost_ratio"]
                self.assertAlmostEqual(expected, measured, places=9)
                self.assertEqual(
                    official["tiers"][tier]["near_budget"],
                    stress.whole_split_tier(fixture, tier)["near_budget"],
                )


@unittest.skipUnless(
    DEV_INPUT.is_file() and TRAIN_INPUT.is_file(),
    "공개 자료가 없습니다: tools/materialize_public_data.py를 먼저 실행하십시오.",
)
class StressPublicSplitTest(unittest.TestCase):
    """Acceptance gates for the deterministic 500-resample composition stress.

    Run the same audit by hand with::

        PYTHONPATH=src python3 tools/stress_safe_margin.py --split dev
        PYTHONPATH=src python3 tools/stress_safe_margin.py --split train
    """

    #: Realized whole-split cost ratio ceilings on public Dev.
    WHOLE_SPLIT_CEILINGS = {"fast": 1.18, "balanced": 1.80, "premium": 3.50}
    #: 500-resample p99 ceilings on public Dev.
    DEV_P99_CEILINGS = {"fast": 1.23, "balanced": 1.90, "premium": 3.75}

    @classmethod
    def setUpClass(cls):
        cls.reports = {
            split: stress.stress_split(
                split=split,
                resamples=stress.DEFAULT_RESAMPLES,
                seed=stress.DEFAULT_SEED,
            )
            for split in ("dev", "train")
        }

    def test_no_resample_breaches_a_hard_cap_on_either_split(self):
        for split, report in self.reports.items():
            for tier in TIERS:
                with self.subTest(split=split, tier=tier):
                    entry = report["bootstrap"][tier]
                    self.assertEqual(0, entry["cap_breaches"])
                    self.assertLessEqual(
                        entry["cost_ratio"]["max"], entry["budget_multiplier"]
                    )

    def test_the_dev_resample_tail_stays_under_the_retune_targets(self):
        report = self.reports["dev"]
        for tier, ceiling in self.DEV_P99_CEILINGS.items():
            with self.subTest(tier=tier):
                self.assertLessEqual(
                    report["bootstrap"][tier]["cost_ratio"]["p99"], ceiling
                )

    def test_the_dev_whole_split_keeps_its_margin_and_no_warning_flag(self):
        report = self.reports["dev"]
        for tier, ceiling in self.WHOLE_SPLIT_CEILINGS.items():
            with self.subTest(tier=tier):
                entry = report["whole_split"][tier]
                self.assertLessEqual(entry["cost_ratio"], ceiling)
                self.assertTrue(entry["budget_passed"])
                self.assertFalse(entry["near_budget"])

    def test_the_train_split_is_audited_with_the_same_configuration(self):
        for split, report in self.reports.items():
            with self.subTest(split=split):
                self.assertEqual(stress.DEFAULT_SEED, report["seed"])
                self.assertEqual(stress.DEFAULT_RESAMPLES, report["resamples"])
                self.assertEqual(split, report["split"])

    def test_the_think_model_stays_inside_premium_under_resampling(self):
        for split, report in self.reports.items():
            for tier in ("fast", "balanced"):
                with self.subTest(split=split, tier=tier):
                    counts = report["whole_split"][tier]["model_counts"]
                    self.assertEqual(0, counts[MODEL_IDS[2]])

    def test_the_measured_tail_is_reported_rather_than_assumed_away(self):
        # The guards reduce the realized tail, they do not remove it. Keeping
        # this assertion honest stops the report from being read as a proof.
        premium = self.reports["dev"]["whole_split"]["premium"]
        multiples = premium["selected_step_multiple"][MODEL_IDS[2]]
        self.assertGreater(multiples["max"], 10.0)
        self.assertTrue(math.isfinite(multiples["max"]))


if __name__ == "__main__":
    unittest.main()
