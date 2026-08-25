# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the risk-calibrated v2 candidate router."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    ProtocolError,
    load_bundled_policy,
    parse_input,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "baselines/risk-calibrated-public.v2.json"
LIGHT_ID, AX31_ID, THINK_ID = MODEL_IDS


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hash_regex = _load_module("hash_regex", ROOT / "baselines/hash_regex.py")
safe_margin = _load_module("safe_margin", ROOT / "baselines/safe_margin.py")
risk_calibrated = _load_module(
    "risk_calibrated", ROOT / "baselines/risk_calibrated.py"
)


_PROMPTS = (
    "오늘 날씨를 한 문장으로 알려 주세요.",
    "Summarize this paragraph in two sentences. It is a short note about trains.",
    (
        "Prove that x^2 + 2*x + 1 = (x+1)^2 by induction and derive every "
        "intermediate step. Numbers: 12, 24, 48, 96, 128, 256."
    ),
    (
        "Given the traceback below, explain the exception and the time "
        "complexity of the fix.\n```python\ndef solve(values):\n"
        "    return sorted(values)\n```"
    ),
    "번역해 주세요: The quick brown fox jumps over the lazy dog.",
    (
        "Answer exactly one of A or B. You must use at least three constraints "
        "and must not use any external source. 정확히 하나만 고르십시오."
    ),
)


def _batch(prefix="episode", reverse=False, challenge_id="v2-test", split="synthetic"):
    prompts = list(_PROMPTS)
    order = list(range(len(prompts)))
    if reverse:
        order.reverse()
    return parse_input(
        {
            "schema_version": 1,
            "challenge_id": challenge_id,
            "split": split,
            "episodes": [
                {
                    "episode_id": f"{prefix}-{index:04d}",
                    "prompt": prompts[position],
                }
                for index, position in enumerate(order)
            ],
        }
    )


class ArtifactValidationTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_bundled_policy()
        self.raw = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    def test_bundled_artifact_parses_and_matches_the_policy(self):
        artifact = risk_calibrated.load_artifact(ARTIFACT_PATH)
        self.assertEqual(self.policy.policy_id, artifact.policy_id)
        for tier in TIERS:
            self.assertLess(
                artifact.tier_plans[tier].target_ratio,
                float(self.policy.tiers[tier].budget_multiplier) + 1e-9,
            )

    def test_an_unknown_or_missing_field_is_rejected(self):
        broken = dict(self.raw)
        broken["extra_field"] = 1
        with self.assertRaises(ProtocolError):
            risk_calibrated.parse_artifact(broken)
        broken = dict(self.raw)
        del broken["cost_upper"]
        with self.assertRaises(ProtocolError):
            risk_calibrated.parse_artifact(broken)

    def test_a_non_finite_head_value_is_rejected_at_parse(self):
        broken = json.loads(json.dumps(self.raw))
        broken["gain_heads"]["ax31"]["intercept"] = "NaN"
        with self.assertRaises(ProtocolError):
            risk_calibrated.parse_artifact(broken)

    def test_missing_conformal_coverage_blocks_the_artifact(self):
        # A cost model that misses its declared coverage gate must not
        # authorize additional budget use: validation has to fail closed.
        broken = json.loads(json.dumps(self.raw))
        declared = float(broken["cost_upper"]["declared_coverage"])
        broken["cost_upper"]["measured_coverage"][AX31_ID] = declared - 0.1
        with self.assertRaises(ProtocolError):
            risk_calibrated.parse_artifact(broken)

    def test_a_wrong_policy_digest_blocks_the_submission(self):
        broken = json.loads(json.dumps(self.raw))
        broken["policy_sha256"] = "0" * 64
        artifact = risk_calibrated.parse_artifact(broken)
        with self.assertRaises(ProtocolError):
            risk_calibrated.make_risk_calibrated_submission(
                _batch(), self.policy, artifact, "fast"
            )

    def test_probability_calibration_rejects_non_monotone_maps(self):
        broken = json.loads(json.dumps(self.raw))
        broken["gain_probability_calibration"]["ax31"] = {
            "method": "isotonic",
            "thresholds": [0.0, 1.0],
            "values": [0.8, 0.2],
        }
        with self.assertRaises(ProtocolError):
            risk_calibrated.parse_artifact(broken)


class PlannerBehaviorTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_bundled_policy()
        self.artifact = risk_calibrated.load_artifact(ARTIFACT_PATH)

    def test_every_episode_is_decided_exactly_once_for_every_tier(self):
        inputs = _batch()
        for tier in TIERS:
            plan = risk_calibrated.make_risk_calibrated_submission(
                inputs, self.policy, self.artifact, tier
            )
            decided = [item.episode_id for item in plan.submission.decisions]
            self.assertEqual(
                sorted(episode.episode_id for episode in inputs.episodes),
                sorted(decided),
            )
            self.assertLessEqual(
                plan.predicted_budget_ratio,
                min(
                    self.artifact.tier_plans[tier].target_ratio,
                    float(self.policy.tiers[tier].budget_multiplier),
                )
                + 1e-9,
            )

    def test_decisions_ignore_episode_ids_order_and_metadata(self):
        base = _batch()
        renamed = _batch(prefix="other", reverse=True, challenge_id="renamed")
        for tier in TIERS:
            first = risk_calibrated.make_risk_calibrated_submission(
                base, self.policy, self.artifact, tier
            )
            second = risk_calibrated.make_risk_calibrated_submission(
                renamed, self.policy, self.artifact, tier
            )
            by_prompt_first = {
                episode.prompt: decision.model_id
                for episode, decision in zip(
                    base.episodes, first.submission.decisions
                )
            }
            by_prompt_second = {
                episode.prompt: decision.model_id
                for episode, decision in zip(
                    renamed.episodes, second.submission.decisions
                )
            }
            self.assertEqual(by_prompt_first, by_prompt_second)

    def test_repeated_planning_is_deterministic(self):
        inputs = _batch()
        for tier in TIERS:
            first = risk_calibrated.make_risk_calibrated_submission(
                inputs, self.policy, self.artifact, tier
            )
            second = risk_calibrated.make_risk_calibrated_submission(
                inputs, self.policy, self.artifact, tier
            )
            self.assertEqual(
                [item.model_id for item in first.submission.decisions],
                [item.model_id for item in second.submission.decisions],
            )

    def test_think_model_is_restricted_to_premium(self):
        inputs = _batch()
        for tier in ("fast", "balanced"):
            plan = risk_calibrated.make_risk_calibrated_submission(
                inputs, self.policy, self.artifact, tier
            )
            self.assertEqual(0, plan.model_counts[THINK_ID])

    def test_runtime_prediction_path_never_sees_identifiers(self):
        episode = _batch().episodes[0]
        prediction = risk_calibrated.predict_episode(
            episode, self.artifact, self.policy
        )
        self.assertNotIn("episode_id", str(prediction))

    def test_upper_costs_dominate_mean_costs_and_stay_monotone(self):
        for episode in _batch().episodes:
            prediction = risk_calibrated.predict_episode(
                episode, self.artifact, self.policy
            )
            previous_mean = 0.0
            previous_upper = 0.0
            for model_id in MODEL_IDS:
                self.assertGreaterEqual(
                    prediction.upper_costs[model_id],
                    prediction.costs[model_id] - 1e-12,
                )
                self.assertGreaterEqual(
                    prediction.costs[model_id], previous_mean
                )
                self.assertGreaterEqual(
                    prediction.upper_costs[model_id], previous_upper
                )
                previous_mean = prediction.costs[model_id]
                previous_upper = prediction.upper_costs[model_id]

    def test_mean_costs_are_floored_by_the_public_rate_ratio(self):
        for episode in _batch().episodes:
            prediction = risk_calibrated.predict_episode(
                episode, self.artifact, self.policy
            )
            light = prediction.costs[LIGHT_ID]
            for model_id in MODEL_IDS[1:]:
                floor = light * risk_calibrated._rate_ratio(
                    self.policy, model_id
                )
                self.assertGreaterEqual(
                    prediction.costs[model_id], floor * (1 - 1e-9)
                )

    def test_plan_group_allocation_ignores_prediction_order(self):
        inputs = _batch()
        predictions = risk_calibrated.predict_batch(
            inputs.episodes, self.artifact, self.policy
        )
        forward, _ratio, _stages = risk_calibrated.plan_selection(
            predictions, self.policy, "balanced", self.artifact.tier_plans
        )
        reversed_predictions = tuple(reversed(predictions))
        backward, _ratio, _stages = risk_calibrated.plan_selection(
            reversed_predictions, self.policy, "balanced", self.artifact.tier_plans
        )
        self.assertEqual(list(forward), list(reversed(backward)))

    def test_rejects_an_unknown_tier_and_an_empty_batch(self):
        predictions = risk_calibrated.predict_batch(
            _batch().episodes, self.artifact, self.policy
        )
        with self.assertRaises(ProtocolError):
            risk_calibrated.plan_selection(
                predictions, self.policy, "ultra", self.artifact.tier_plans
            )
        with self.assertRaises(ValueError):
            risk_calibrated.plan_selection(
                (), self.policy, "fast", self.artifact.tier_plans
            )


class CliAndFallbackTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_bundled_policy()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = pathlib.Path(self.temporary.name)
        self.input_path = self.directory / "inputs.json"
        batch = _batch()
        self.input_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "challenge_id": batch.challenge_id,
                    "split": batch.split,
                    "episodes": [
                        {
                            "episode_id": episode.episode_id,
                            "prompt": episode.prompt,
                        }
                        for episode in batch.episodes
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_cli_writes_byte_identical_output_on_repeated_runs(self):
        first = self.directory / "first.json"
        second = self.directory / "second.json"
        for output in (first, second):
            code = risk_calibrated.main(
                [
                    "--input",
                    str(self.input_path),
                    "--tier",
                    "premium",
                    "--output",
                    str(output),
                    "--artifact",
                    str(ARTIFACT_PATH),
                ]
            )
            self.assertEqual(0, code)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_a_corrupt_artifact_falls_back_to_safe_margin_decisions(self):
        corrupt = self.directory / "artifact.json"
        corrupt.write_text("{\"artifact_type\": \"wrong\"}", encoding="utf-8")
        fallback_output = self.directory / "fallback.json"
        code = risk_calibrated.main(
            [
                "--input",
                str(self.input_path),
                "--tier",
                "balanced",
                "--output",
                str(fallback_output),
                "--artifact",
                str(corrupt),
            ]
        )
        self.assertEqual(0, code)
        reference_output = self.directory / "reference.json"
        code = safe_margin.main(
            [
                "--input",
                str(self.input_path),
                "--tier",
                "balanced",
                "--output",
                str(reference_output),
            ]
        )
        self.assertEqual(0, code)
        self.assertEqual(
            fallback_output.read_bytes(), reference_output.read_bytes()
        )

    def test_no_fallback_flag_surfaces_the_artifact_error(self):
        corrupt = self.directory / "artifact.json"
        corrupt.write_text("not json", encoding="utf-8")
        output = self.directory / "never.json"
        code = risk_calibrated.main(
            [
                "--input",
                str(self.input_path),
                "--tier",
                "fast",
                "--output",
                str(output),
                "--artifact",
                str(corrupt),
                "--no-fallback",
            ]
        )
        self.assertEqual(2, code)
        self.assertFalse(output.exists())

    def test_a_missing_artifact_file_still_produces_a_submission(self):
        output = self.directory / "missing-artifact.json"
        code = risk_calibrated.main(
            [
                "--input",
                str(self.input_path),
                "--tier",
                "fast",
                "--output",
                str(output),
                "--artifact",
                str(self.directory / "does-not-exist.json"),
            ]
        )
        self.assertEqual(0, code)
        self.assertTrue(output.exists())


class RuntimeIsolationTest(unittest.TestCase):
    """The candidate runtime must stay outcome-free and tool-free."""

    def test_runtime_modules_never_import_development_tools(self):
        for relative in (
            "baselines/risk_calibrated.py",
            "baselines/safe_margin.py",
            "baselines/hash_regex.py",
            "container/entrypoint.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            # Documentation may *mention* the development tools; importing
            # them (or the outcome loader) from the runtime path may not
            # happen in any form.
            for forbidden in (
                "import oracle_headroom",
                "from oracle_headroom",
                "import risk_validation",
                "from risk_validation",
                "import run_mvp",
                "from run_mvp",
                "import stress_safe_margin",
                "from stress_safe_margin",
                "load_outcomes",
            ):
                self.assertNotIn(
                    forbidden,
                    text,
                    f"{relative} must not reference {forbidden}",
                )

    def test_submitted_entrypoint_still_runs_the_safe_margin_policy(self):
        # The champion gates were not met, so the submitted container path
        # must keep executing the safe-margin policy unchanged.
        entrypoint = (ROOT / "container/entrypoint.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('ROUTER_MODULE_NAME = "safe_margin"', entrypoint)
        dockerfile = (ROOT / "container/Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("risk_calibrated", dockerfile)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
