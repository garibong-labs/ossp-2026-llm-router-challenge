# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the frozen Premium think-loss veto experiment."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "baselines", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import risk_validation
import safe_margin
import safe_margin_think_loss_veto_runtime as runtime
import representation_features
from ossp_router.protocol import (
    MODEL_IDS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    Submission,
    load_bundled_policy,
    policy_sha256,
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "safe_margin_think_loss_veto_tool", ROOT / "tools/safe_margin_think_loss_veto.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _artifact(votes: int, threshold: int = 7):
    heads = []
    for head_index in range(9):
        indices = tool.head_feature_indices(head_index)
        heads.append(runtime.VetoHead(
            indices,
            tuple(0.0 for _ in indices),
            tuple(1.0 for _ in indices),
            -1.0 if head_index < votes else 1.0,
            tuple(0.0 for _ in indices),
            0.0,
        ))
    return runtime.VetoArtifact(tuple(heads), threshold, "policy", "digest")


def _fixture(models=(MODEL_IDS[2], MODEL_IDS[2], MODEL_IDS[1])):
    episodes = tuple(Episode(f"e{index}", prompt="Question: same shape?") for index in range(len(models)))
    inputs = InputBatch(1, "challenge", "synthetic", episodes)
    submission = Submission(1, "challenge", "policy", "synthetic", "premium", tuple(
        Decision(episode.episode_id, model) for episode, model in zip(episodes, models)
    ))
    baseline = safe_margin.SafeMarginPlan(submission, 2.0, 3.2, {}, ())
    predictions = tuple(safe_margin.EpisodePrediction(
        {MODEL_IDS[0]: 0.0, MODEL_IDS[1]: 0.2, MODEL_IDS[2]: 0.3},
        {MODEL_IDS[0]: 1.0, MODEL_IDS[1]: 2.0, MODEL_IDS[2]: 3.0},
        (0,) if index < 2 else (1,),
    ) for index in range(len(models)))
    return inputs, baseline, predictions


class ConsensusAndRoutingTest(unittest.TestCase):
    def test_training_and_runtime_share_all_three_safe_margin_additions(self):
        inputs, _baseline, predictions = _fixture()
        vectors = runtime.feature_vectors(inputs, predictions)
        structural_count = len(
            representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES
        )
        self.assertEqual(36, structural_count)
        self.assertEqual(39, len(vectors[0]))
        self.assertAlmostEqual(0.1, vectors[0][structural_count])
        self.assertEqual((1.0, -4.0), vectors[0][structural_count + 1:])
        training_matrix = np.asarray(
            runtime.feature_vectors(inputs, predictions), dtype=np.float64
        )
        self.assertEqual(tuple(vectors[0]), tuple(training_matrix[0]))
        for head in range(9):
            indices = tool.head_feature_indices(head)
            self.assertEqual(35, len(indices))
            self.assertEqual((36, 37, 38), indices[-3:])
            self.assertEqual(
                tuple(index for index in range(36) if index % 9 != head),
                indices[:-3],
            )

    def test_exact_consensus_thresholds_and_negative_bound_semantics(self):
        inputs, baseline, predictions = _fixture()
        seven = runtime.apply_veto(inputs, baseline, predictions, _artifact(7, 7))
        eight = runtime.apply_veto(inputs, baseline, predictions, _artifact(7, 8))
        self.assertEqual((MODEL_IDS[1], MODEL_IDS[1], MODEL_IDS[1]), tuple(item.model_id for item in seven.submission.decisions))
        self.assertEqual(tuple(item.model_id for item in baseline.submission.decisions), tuple(item.model_id for item in eight.submission.decisions))
        tied = _artifact(9, 9)
        tied_heads = tuple(runtime.VetoHead(head.feature_indices, head.mean, head.scale, 0.0, head.coefficients, 0.0) for head in tied.heads)
        tied = runtime.VetoArtifact(tied_heads, 9, "policy", "digest")
        self.assertEqual(0, runtime.apply_veto(inputs, baseline, predictions, tied).episodes_vetoed)

    def test_veto_is_group_atomic_and_leaves_non_think_premium_unchanged(self):
        inputs, baseline, predictions = _fixture()
        plan = runtime.apply_veto(inputs, baseline, predictions, _artifact(9, 9))
        models = tuple(item.model_id for item in plan.submission.decisions)
        self.assertEqual((MODEL_IDS[1], MODEL_IDS[1], MODEL_IDS[1]), models)
        self.assertEqual(1, plan.groups_vetoed)
        self.assertEqual(2, plan.episodes_vetoed)

    def test_fast_and_balanced_are_byte_for_byte_unchanged(self):
        inputs, baseline, predictions = _fixture()
        for tier in ("fast", "balanced"):
            submission = Submission(
                1, "challenge", "policy", "synthetic", tier,
                baseline.submission.decisions,
            )
            unchanged = safe_margin.SafeMarginPlan(submission, 1.5, 1.6, {}, ())
            plan = runtime.apply_veto(
                inputs, unchanged, predictions, _artifact(9, 9)
            )
            self.assertEqual(unchanged.submission, plan.submission)

    def test_post_processing_cannot_refill_budget_or_upgrade(self):
        inputs, baseline, predictions = _fixture()
        before = tuple(item.model_id for item in baseline.submission.decisions)
        after = tuple(item.model_id for item in runtime.apply_veto(inputs, baseline, predictions, _artifact(9, 9)).submission.decisions)
        allowed = {(MODEL_IDS[2], MODEL_IDS[1]), (MODEL_IDS[1], MODEL_IDS[1])}
        self.assertTrue(all(pair in allowed for pair in zip(before, after)))


class HonestyTest(unittest.TestCase):
    def test_nested_predictions_are_family_and_group_disjoint(self):
        families = tuple(family for family in risk_validation.FAMILY_LABELS for _ in range(2))
        rows = len(families)
        matrix = np.arange(rows * 39, dtype=float).reshape(rows, 39) / 100.0
        targets = np.linspace(-0.2, 0.2, rows)
        keys = tuple((index,) for index in range(rows))
        heads, audit = tool.fit_outer_heads(matrix, targets, keys, families, families[0])
        self.assertEqual(9, len(heads))
        for head in audit["heads"]:
            self.assertTrue(head["outer_family_excluded"])
            self.assertTrue(head["outer_groups_purged"])
            self.assertTrue(all(row["family_disjoint"] and row["group_disjoint"] for row in head["inner"]))

    def test_cross_boundary_group_is_purged_from_fit(self):
        families = tuple(family for family in risk_validation.FAMILY_LABELS for _ in range(2))
        matrix = np.arange(len(families) * 39, dtype=float).reshape(len(families), 39)
        targets = np.zeros(len(families))
        keys = [(index,) for index in range(len(families))]
        keys[0] = keys[2]
        heads, audit = tool.fit_outer_heads(matrix, targets, tuple(keys), families, families[0])
        self.assertEqual(9, len(heads))
        self.assertTrue(all(head["outer_groups_purged"] for head in audit["heads"]))


class ArtifactAndReportTest(unittest.TestCase):
    def test_artifact_identity_requires_fixed_additions_in_every_head(self):
        policy = load_bundled_policy()
        heads = []
        for head_index in range(9):
            indices = list(tool.head_feature_indices(head_index))
            heads.append({
                "head": head_index,
                "feature_indices": indices,
                "mean": [0.0] * len(indices),
                "scale": [1.0] * len(indices),
                "intercept": 0.0,
                "coefficients": [0.0] * len(indices),
                "upper_residual": 0.0,
            })
        artifact = {
            "artifact_type": runtime.ARTIFACT_TYPE,
            "schema_version": 1,
            "protocol_sha256": runtime.PROTOCOL_SHA256,
            "base_commit": runtime.BASE_COMMIT,
            "feature_version": runtime.FEATURE_VERSION,
            "feature_names": list(runtime.FEATURE_NAMES),
            "heads": heads,
            "minimum_negative_votes": 7,
            "policy_id": policy.policy_id,
            "policy_sha256": policy_sha256(policy),
            "training_data_sha256": "0" * 64,
        }
        parsed = runtime.parse_artifact(artifact)
        self.assertTrue(all(len(head.feature_indices) == 35 for head in parsed.heads))
        artifact["feature_names"] = artifact["feature_names"][:-1]
        with self.assertRaises(ProtocolError):
            runtime.parse_artifact(artifact)

    def test_missing_and_corrupt_artifacts_fail_closed(self):
        policy = load_bundled_policy()
        safe_artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
        inputs = InputBatch(1, "challenge", "synthetic", (Episode("e", prompt="hello"),))
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            plan = runtime.make_submission(inputs, policy, safe_artifact, "premium", missing)
            self.assertFalse(plan.artifact_valid)
            corrupt = Path(directory) / "corrupt.json"
            corrupt.write_text("{", encoding="utf-8")
            plan = runtime.make_submission(inputs, policy, safe_artifact, "premium", corrupt)
            self.assertFalse(plan.artifact_valid)

    def test_protocol_hash_and_rejected_report_gate_integrity_are_reproducible(self):
        self.assertEqual(runtime.PROTOCOL_SHA256, tool.verify_protocol())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = tool._parser().parse_args([
                "--train-input", str(root / "missing-train.json"),
                "--train-outcomes", str(ROOT / "data/train/outcomes.json"),
                "--dev-input", str(root / "missing-dev.json"),
                "--dev-outcomes", str(ROOT / "data/dev/outcomes.json"),
                "--artifact", str(root / "artifact.json"),
                "--report", str(root / "report.json"),
            ])
            first_artifact, first_report = tool.run(args)
            second_artifact, second_report = tool.run(args)
            self.assertEqual(first_artifact, second_artifact)
            self.assertEqual(first_report, second_report)
            self.assertFalse(first_report["dev"]["accessed"])
            self.assertFalse(first_report["safety"]["accessed"])
            self.assertEqual("safe-margin", first_report["submission_default"])
            self.assertTrue(all(not item["gate_passed"] for item in first_report["candidate_train_metrics"].values()))


if __name__ == "__main__":
    unittest.main()
