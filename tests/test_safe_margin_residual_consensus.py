# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused contract checks for residual consensus v1."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "baselines", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import representation_features  # noqa: E402
import run_safe_margin_residual_consensus as runner  # noqa: E402
import safe_margin  # noqa: E402
import safe_margin_residual_consensus as candidate  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    Episode,
    InputBatch,
    load_bundled_policy,
)


class FrozenProtocolTest(unittest.TestCase):
    def test_contract_hash_and_candidate_grid_are_frozen(self):
        digest = hashlib.sha256(runner.PROTOCOL_PATH.read_bytes()).hexdigest()
        self.assertEqual(runner.EXPECTED_PROTOCOL_SHA256, digest)
        protocol = runner.validate_protocol()
        strengths = protocol["candidate"]["correction_strengths"]
        self.assertEqual([0.25, 0.5, 1.0], strengths)
        self.assertLessEqual(len(strengths), 3)

    def test_dev_loader_is_not_reached_when_train_gate_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = pathlib.Path(directory) / "report.json"
            with mock.patch.object(
                runner,
                "evaluate_train",
                return_value={"gate_passed": False},
            ), mock.patch.object(
                runner,
                "evaluate_dev_once",
                side_effect=AssertionError("Dev must remain closed"),
            ):
                report = runner.run_experiment(report_path=report_path)
        self.assertFalse(report["dev"]["gate_opened"])
        self.assertFalse(report["dev"]["dev_outcomes_opened"])
        self.assertEqual("safe-margin", report["adoption"]["submission_default"])

    def test_atomic_report_regeneration_is_byte_stable(self):
        value = {"z": [0.25, 0.5, 1.0], "a": {"passed": False}}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            runner._write_json_atomic(path, value)
            first = path.read_bytes()
            runner._write_json_atomic(path, value)
            self.assertEqual(first, path.read_bytes())


def _prediction(gain, increment, signature):
    light, ax31, think = MODEL_IDS
    return safe_margin.EpisodePrediction(
        scores={light: 0.5, ax31: 0.5 + gain, think: 0.9},
        costs={light: 1.0, ax31: 1.0 + increment, think: 2.0},
        signature=(signature,),
    )


class ExchangeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()

    def test_fast_and_balanced_never_exceed_matched_safe_spend(self):
        predictions = tuple(
            _prediction(gain, increment, index)
            for index, (gain, increment) in enumerate(
                ((0.08, 0.20), (0.04, 0.20), (0.03, 0.20), (0.02, 0.20))
            )
        )
        residuals = (-0.1, 0.1, 0.1, 0.1)
        for tier in ("fast", "balanced"):
            selected, ratio, matched = candidate.plan_selection(
                predictions, residuals, self.policy, tier, 1.0
            )
            self.assertLessEqual(ratio, matched + 1e-12)
            safe_selected = safe_margin.plan_selection(
                predictions, self.policy, tier
            )[0]
            eligible = {
                index
                for index, item in enumerate(predictions)
                if item.scores[MODEL_IDS[1]] - item.scores[MODEL_IDS[0]]
                >= safe_margin.TIER_PLAN_CONFIGS[tier].ax31_min_gain
            }
            self.assertTrue(
                all(
                    model_id == MODEL_IDS[0] or index in eligible
                    for index, model_id in enumerate(selected)
                )
            )
            self.assertEqual(len(selected), len(safe_selected))

    def test_premium_and_think_decisions_are_exact_safe_margin_identity(self):
        predictions = tuple(
            _prediction(0.2, 0.2, index) for index in range(8)
        )
        safe = safe_margin.plan_selection(predictions, self.policy, "premium")
        measured = candidate.plan_selection(
            predictions,
            (0.1,) * len(predictions),
            self.policy,
            "premium",
            1.0,
        )
        self.assertEqual(safe[0], measured[0])
        self.assertEqual(safe[1], measured[1])


class RuntimeIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()
        cls.base_artifact = safe_margin.load_artifact(
            safe_margin.DEFAULT_ARTIFACT_PATH
        )

    def test_features_ignore_episode_identity_and_enforce_field_bound(self):
        bound = representation_features.MAX_FIELD_CHARACTERS
        prefix = "Question: x\n" + "a" * (bound - len("Question: x\n"))
        left = Episode("family-source-row-dev-1", prompt=prefix + "alpha")
        right = Episode("train-999", prompt=prefix + "beta")
        self.assertEqual(
            representation_features.expanded_structural_vector(left),
            representation_features.expanded_structural_vector(right),
        )
        self.assertEqual(32768, bound)

    def test_feature_extraction_does_not_read_episode_identity(self):
        class PromptOnlyView:
            prompt = "Question: Which structural shape is present?"
            messages = None

            @property
            def episode_id(self):
                raise AssertionError("episode_id reached the runtime feature path")

        vector = representation_features.expanded_structural_vector(
            PromptOnlyView()
        )
        self.assertEqual(36, len(vector))

    def test_corrupt_artifact_falls_back_to_safe_margin(self):
        inputs = InputBatch(
            schema_version=1,
            challenge_id="fallback-test",
            split="synthetic",
            episodes=(Episode("one", prompt="Explain 2 + 2."),),
        )
        with tempfile.TemporaryDirectory() as directory:
            corrupt = pathlib.Path(directory) / "artifact.json"
            corrupt.write_text("{}\n", encoding="utf-8")
            plan = candidate.make_submission_or_safe_margin(
                inputs,
                self.policy,
                self.base_artifact,
                "premium",
                artifact_path=corrupt,
            )
        safe = safe_margin.make_safe_margin_submission(
            inputs, self.policy, self.base_artifact, "premium"
        )
        self.assertTrue(plan.fallback_used)
        self.assertEqual(safe.submission, plan.submission)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
