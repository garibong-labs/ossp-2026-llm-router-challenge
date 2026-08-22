# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the safe-margin MVP router."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import stat
import sys
import tempfile
import unittest

from ossp_router.heuristic import episode_text
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_outcomes,
    parse_input,
    submission_to_dict,
)
from ossp_router.scoring import score_submissions


ROOT = pathlib.Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "baselines/hash-regex-public.v1.json"
LIGHT_ID, AX31_ID, THINK_ID = MODEL_IDS


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hash_regex = _load_module("hash_regex", ROOT / "baselines/hash_regex.py")
safe_margin = _load_module("safe_margin", ROOT / "baselines/safe_margin.py")


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


def _batch(prefix="episode", reverse=False, challenge_id="mvp-test", split="synthetic"):
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
                {"episode_id": f"{prefix}-{position}", "prompt": prompts[index]}
                for position, index in enumerate(order)
            ],
        }
    )


def _decisions_by_content(inputs, submission):
    model_by_id = {
        decision.episode_id: decision.model_id for decision in submission.decisions
    }
    return {
        episode_text(episode): model_by_id[episode.episode_id]
        for episode in inputs.episodes
    }


def _prediction(scores, costs, signature=(0,)):
    return safe_margin.EpisodePrediction(
        scores=dict(scores), costs=dict(costs), signature=tuple(signature)
    )


class SafeMarginOutputTest(unittest.TestCase):
    """Schema completeness, determinism and identity/order invariance."""

    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()
        cls.artifact = hash_regex.load_artifact(ARTIFACT_PATH)

    def _plan(self, inputs, tier):
        return safe_margin.make_safe_margin_submission(
            inputs, self.policy, self.artifact, tier
        )

    def test_every_episode_is_decided_exactly_once_for_every_tier(self):
        inputs = _batch()
        for tier in TIERS:
            with self.subTest(tier=tier):
                plan = self._plan(inputs, tier)
                submission = plan.submission
                self.assertEqual(tier, submission.tier)
                self.assertEqual(inputs.challenge_id, submission.challenge_id)
                self.assertEqual(inputs.split, submission.split)
                self.assertEqual(self.policy.policy_id, submission.policy_id)
                self.assertEqual(inputs.schema_version, submission.schema_version)
                decided = [decision.episode_id for decision in submission.decisions]
                self.assertEqual(
                    sorted(episode.episode_id for episode in inputs.episodes),
                    sorted(decided),
                )
                self.assertEqual(len(decided), len(set(decided)))
                for decision in submission.decisions:
                    self.assertIn(decision.model_id, MODEL_IDS)
                self.assertEqual(
                    len(inputs.episodes), sum(plan.model_counts.values())
                )

    def test_repeated_runs_produce_identical_bytes(self):
        inputs = _batch()
        for tier in TIERS:
            with self.subTest(tier=tier):
                first = submission_to_dict(self._plan(inputs, tier).submission)
                second = submission_to_dict(self._plan(inputs, tier).submission)
                self.assertEqual(
                    json.dumps(first, sort_keys=True),
                    json.dumps(second, sort_keys=True),
                )

    def test_episode_id_order_and_metadata_do_not_change_decisions(self):
        original = _batch()
        shuffled = _batch(
            prefix="totally-different",
            reverse=True,
            challenge_id="other-challenge",
            split="other-split",
        )
        for tier in TIERS:
            with self.subTest(tier=tier):
                first = _decisions_by_content(
                    original, self._plan(original, tier).submission
                )
                second = _decisions_by_content(
                    shuffled, self._plan(shuffled, tier).submission
                )
                self.assertEqual(first, second)

    def test_router_never_reads_identifiers_from_the_prediction_path(self):
        inputs = _batch()
        renamed = _batch(prefix="zzz")
        first = safe_margin.predict_batch(inputs.episodes, self.artifact, self.policy)
        second = safe_margin.predict_batch(
            renamed.episodes, self.artifact, self.policy
        )
        self.assertEqual(first, second)

    def test_predict_from_raw_matches_the_official_predict_episode(self):
        for episode in _batch().episodes:
            with self.subTest(episode=episode.episode_id):
                raw = safe_margin.raw_feature_vector(
                    episode, self.artifact.hash_bins
                )
                scores, costs = safe_margin.predict_from_raw(raw, self.artifact)
                expected_scores, expected_costs = hash_regex.predict_episode(
                    episode, self.artifact
                )
                self.assertEqual(dict(expected_scores), scores)
                self.assertEqual(dict(expected_costs), costs)

    def test_rejects_an_artifact_that_does_not_match_the_policy(self):
        broken = dataclasses.replace(self.artifact, policy_digest="0" * 64)
        with self.assertRaises(ProtocolError):
            safe_margin.make_safe_margin_submission(
                _batch(), self.policy, broken, "fast"
            )


class SafeMarginTierSeparationTest(unittest.TestCase):
    """Tier separation and the `axk1-think` restriction."""

    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()
        cls.artifact = hash_regex.load_artifact(ARTIFACT_PATH)
        cls.inputs = _batch()
        cls.plans = {
            tier: safe_margin.make_safe_margin_submission(
                cls.inputs, cls.policy, cls.artifact, tier
            )
            for tier in TIERS
        }

    def test_predicted_cost_ratio_increases_with_the_tier(self):
        ratios = [self.plans[tier].predicted_budget_ratio for tier in TIERS]
        self.assertLessEqual(ratios[0], ratios[1])
        self.assertLessEqual(ratios[1], ratios[2])

    def test_think_model_is_restricted_to_premium(self):
        for tier in ("fast", "balanced"):
            with self.subTest(tier=tier):
                self.assertEqual(0, self.plans[tier].model_counts[THINK_ID])
                self.assertFalse(
                    safe_margin.TIER_PLAN_CONFIGS[tier].allow_think
                )
        self.assertTrue(safe_margin.TIER_PLAN_CONFIGS["premium"].allow_think)

    def test_configured_targets_stay_under_the_published_caps(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                config = safe_margin.TIER_PLAN_CONFIGS[tier]
                cap = float(self.policy.tiers[tier].budget_multiplier)
                self.assertLess(config.target_ratio, cap)

    def test_planned_ratio_never_exceeds_the_tier_target(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                config = safe_margin.TIER_PLAN_CONFIGS[tier]
                self.assertLessEqual(
                    self.plans[tier].predicted_budget_ratio,
                    config.target_ratio + 1e-9,
                )


class SafeMarginBudgetGuardTest(unittest.TestCase):
    """Cost model, quality margin, tail guard and group allocation."""

    def setUp(self):
        self.policy = load_bundled_policy()

    def test_conservative_costs_are_floored_by_the_public_rate_ratio(self):
        # A learned head that badly understates ax31 and think must still be
        # floored by the published input-token rate ratio, then inflated.
        raw = {LIGHT_ID: 1.0, AX31_ID: 1.0000001, THINK_ID: 1.0000002}
        costs = safe_margin.conservative_costs(raw, self.policy)
        light_rate = float(self.policy.models[LIGHT_ID].input_token_rate)
        for model_id in (AX31_ID, THINK_ID):
            expected = (
                float(self.policy.models[model_id].input_token_rate) / light_rate
            ) * safe_margin.COST_INFLATION[model_id]
            self.assertAlmostEqual(expected, costs[model_id], places=9)
        self.assertLess(costs[LIGHT_ID], costs[AX31_ID])
        self.assertLess(costs[AX31_ID], costs[THINK_ID])

    def test_conservative_costs_keep_the_upgrade_ladder_monotone(self):
        raw = {LIGHT_ID: 5.0, AX31_ID: 400.0, THINK_ID: 6.0}
        costs = safe_margin.conservative_costs(raw, self.policy)
        self.assertLessEqual(costs[LIGHT_ID], costs[AX31_ID])
        self.assertLessEqual(costs[AX31_ID], costs[THINK_ID])

    def test_conservative_costs_reject_a_non_positive_light_estimate(self):
        with self.assertRaises(ValueError):
            safe_margin.conservative_costs(
                {LIGHT_ID: 0.0, AX31_ID: 1.0, THINK_ID: 2.0}, self.policy
            )

    def test_a_gain_below_the_quality_margin_stays_on_the_cheaper_model(self):
        config = dataclasses.replace(
            safe_margin.TIER_PLAN_CONFIGS["fast"],
            target_ratio=1.25,
            ax31_min_gain=0.05,
        )
        # Predicted gain of 0.01 is real but far below the 0.05 margin.
        predictions = [
            _prediction(
                {LIGHT_ID: 0.50, AX31_ID: 0.51, THINK_ID: 0.52},
                {LIGHT_ID: 1.0, AX31_ID: 1.1, THINK_ID: 2.0},
                signature=(index,),
            )
            for index in range(4)
        ]
        selected, ratio, stages = safe_margin.plan_selection(
            predictions, self.policy, "fast", config
        )
        self.assertEqual(tuple([LIGHT_ID] * 4), selected)
        self.assertEqual(1.0, ratio)
        self.assertEqual(0, stages[0].eligible)

    def test_a_tail_cost_upgrade_is_rejected_even_with_a_large_gain(self):
        config = dataclasses.replace(
            safe_margin.TIER_PLAN_CONFIGS["fast"],
            target_ratio=1.25,
            ax31_min_gain=0.0,
            ax31_max_step_ratio=3.0,
        )
        # ax31 would cost 40x its own light cost: exactly the heavy tail that
        # makes a realized budget overshoot on an unseen prompt mix.
        predictions = [
            _prediction(
                {LIGHT_ID: 0.10, AX31_ID: 0.99, THINK_ID: 0.99},
                {LIGHT_ID: 1.0, AX31_ID: 40.0, THINK_ID: 60.0},
                signature=(index,),
            )
            for index in range(4)
        ]
        selected, ratio, stages = safe_margin.plan_selection(
            predictions, self.policy, "fast", config
        )
        self.assertEqual(tuple([LIGHT_ID] * 4), selected)
        self.assertEqual(1.0, ratio)
        self.assertEqual(4, stages[0].considered)
        self.assertEqual(0, stages[0].eligible)

    def test_an_exhausted_budget_keeps_every_episode_on_the_light_model(self):
        config = dataclasses.replace(
            safe_margin.TIER_PLAN_CONFIGS["fast"], target_ratio=1.0
        )
        predictions = [
            _prediction(
                {LIGHT_ID: 0.10, AX31_ID: 0.90, THINK_ID: 0.95},
                {LIGHT_ID: 1.0, AX31_ID: 2.2, THINK_ID: 7.0},
                signature=(index,),
            )
            for index in range(4)
        ]
        selected, ratio, _stages = safe_margin.plan_selection(
            predictions, self.policy, "fast", config
        )
        self.assertEqual(tuple([LIGHT_ID] * 4), selected)
        self.assertEqual(1.0, ratio)

    def test_a_content_group_is_promoted_as_a_whole_or_not_at_all(self):
        # Four identical-content episodes share one group. The budget fits
        # three of them, so a per-episode greedy would promote three and break
        # the tie by input position. Group allocation must promote none.
        config = dataclasses.replace(
            safe_margin.TIER_PLAN_CONFIGS["balanced"],
            target_ratio=1.90,
            ax31_min_gain=0.0,
            ax31_max_step_ratio=100.0,
        )
        predictions = [
            _prediction(
                {LIGHT_ID: 0.10, AX31_ID: 0.90, THINK_ID: 0.95},
                {LIGHT_ID: 1.0, AX31_ID: 2.2, THINK_ID: 7.0},
                signature=(7,),
            )
            for _index in range(4)
        ]
        selected, ratio, stages = safe_margin.plan_selection(
            predictions, self.policy, "balanced", config
        )
        self.assertEqual(1, stages[0].groups_considered)
        self.assertEqual(0, stages[0].groups_promoted)
        self.assertEqual(tuple([LIGHT_ID] * 4), selected)
        self.assertEqual(1.0, ratio)

    def test_group_allocation_is_invariant_to_prediction_order(self):
        predictions = [
            _prediction(
                {LIGHT_ID: 0.10 + 0.01 * index, AX31_ID: 0.90, THINK_ID: 0.95},
                {LIGHT_ID: 1.0 + index, AX31_ID: 2.2 + index, THINK_ID: 9.0 + index},
                signature=(index % 3,),
            )
            for index in range(9)
        ]
        forward, forward_ratio, _first = safe_margin.plan_selection(
            predictions, self.policy, "balanced"
        )
        reversed_predictions = list(reversed(predictions))
        backward, backward_ratio, _second = safe_margin.plan_selection(
            reversed_predictions, self.policy, "balanced"
        )
        self.assertEqual(list(forward), list(reversed(backward)))
        self.assertAlmostEqual(forward_ratio, backward_ratio, places=12)

    def test_think_upgrades_only_start_from_ax31_and_respect_their_share(self):
        config = dataclasses.replace(
            safe_margin.TIER_PLAN_CONFIGS["premium"],
            target_ratio=4.0,
            ax31_min_gain=0.0,
            think_min_gain=0.0,
            think_max_step_ratio=100.0,
            think_budget_share=0.0,
        )
        predictions = [
            _prediction(
                {LIGHT_ID: 0.10, AX31_ID: 0.50, THINK_ID: 0.99},
                {LIGHT_ID: 1.0, AX31_ID: 2.2, THINK_ID: 9.0},
                signature=(index,),
            )
            for index in range(4)
        ]
        selected, _ratio, stages = safe_margin.plan_selection(
            predictions, self.policy, "premium", config
        )
        self.assertNotIn(THINK_ID, selected)
        self.assertEqual("ax31-light->ax31", stages[0].step)
        self.assertEqual("ax31->axk1-think", stages[1].step)
        self.assertEqual(4, stages[1].considered)
        self.assertEqual(0, stages[1].promoted)

    def test_rejects_an_unknown_tier_and_an_empty_batch(self):
        with self.assertRaises(ProtocolError):
            safe_margin.plan_selection([_prediction(
                {LIGHT_ID: 0.1, AX31_ID: 0.2, THINK_ID: 0.3},
                {LIGHT_ID: 1.0, AX31_ID: 2.0, THINK_ID: 3.0},
            )], self.policy, "turbo")
        with self.assertRaises(ValueError):
            safe_margin.plan_selection([], self.policy, "fast")


class SafeMarginCommandLineTest(unittest.TestCase):
    def test_cli_writes_one_readable_submission_per_tier(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary)
            for tier in TIERS:
                output = target / f"{tier}.json"
                code = safe_margin.main(
                    [
                        "--input",
                        str(ROOT / "data/toy/inputs.json"),
                        "--tier",
                        tier,
                        "--artifact",
                        str(ARTIFACT_PATH),
                        "--output",
                        str(output),
                    ]
                )
                self.assertEqual(0, code)
                self.assertTrue(output.is_file())
                self.assertEqual(
                    0o644, stat.S_IMODE(output.stat().st_mode)
                )
                payload = json.loads(output.read_text(encoding="utf-8"))
                self.assertEqual(tier, payload["tier"])

    def test_cli_reports_a_missing_input_without_writing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "missing.json"
            code = safe_margin.main(
                [
                    "--input",
                    str(pathlib.Path(temporary) / "absent.json"),
                    "--tier",
                    "fast",
                    "--artifact",
                    str(ARTIFACT_PATH),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(2, code)
            self.assertFalse(output.exists())


DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"


@unittest.skipUnless(
    DEV_INPUT.is_file(),
    "공개 Dev 자료가 없습니다: tools/materialize_public_data.py를 먼저 실행하십시오.",
)
class SafeMarginPublicDevTest(unittest.TestCase):
    """End-to-end budget targets on the materialized public Dev split."""

    #: Tier safety targets on the realized public Dev cost ratio.
    TARGETS = {"fast": 1.18, "balanced": 1.80, "premium": 3.50}

    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()
        cls.artifact = hash_regex.load_artifact(ARTIFACT_PATH)
        cls.inputs = load_input(DEV_INPUT)
        cls.outcomes = load_outcomes(ROOT / "data/dev/outcomes.json")
        submissions = [
            safe_margin.make_safe_margin_submission(
                cls.inputs, cls.policy, cls.artifact, tier
            ).submission
            for tier in TIERS
        ]
        cls.report = score_submissions(
            cls.inputs, cls.outcomes, submissions, cls.policy
        )

    def test_every_tier_stays_within_the_public_budget(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                self.assertTrue(self.report["tiers"][tier]["budget_passed"])
                self.assertFalse(self.report["tiers"][tier]["near_budget"])

    def test_realized_cost_ratio_meets_the_tier_safety_target(self):
        for tier, target in self.TARGETS.items():
            with self.subTest(tier=tier):
                ratio = float(self.report["tiers"][tier]["budget_ratio"])
                self.assertLessEqual(ratio, target)

    def test_quality_beats_the_all_light_baseline_on_every_tier(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                quality = float(self.report["tiers"][tier]["quality_score"])
                self.assertGreater(quality, 0.619318)
        self.assertGreater(float(self.report["final_score"]), 0.655341)

    def test_think_model_appears_only_in_the_premium_tier(self):
        for tier in ("fast", "balanced"):
            with self.subTest(tier=tier):
                self.assertEqual(
                    0, self.report["tiers"][tier]["model_counts"][THINK_ID]
                )
        self.assertGreater(
            self.report["tiers"]["premium"]["model_counts"][THINK_ID], 0
        )


if __name__ == "__main__":
    unittest.main()
