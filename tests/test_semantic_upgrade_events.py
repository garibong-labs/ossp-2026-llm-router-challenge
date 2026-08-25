# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "baselines", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import semantic_upgrade_events as events  # noqa: E402
from ossp_router.protocol import Episode, Message  # noqa: E402


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "semantic_upgrade_experiment", ROOT / "tools/semantic_upgrade_experiment.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = _load_tool()


class FrozenProtocolTest(unittest.TestCase):
    protocol_path = ROOT / "configs/semantic-upgrade-events-protocol.v1.json"

    def test_registry_is_small_immutable_and_fully_checksummed(self):
        protocol = experiment.load_protocol(self.protocol_path)
        self.assertLessEqual(len(protocol["candidate_encoders"]), 2)
        self.assertEqual(
            "5697a65b0a002a92fe8c4fc9d495303ffff9c7d2",
            protocol["candidate_encoders"][0]["revision"],
        )
        self.assertEqual("MIT", protocol["candidate_encoders"][0]["license"])
        self.assertTrue(all(len(item["sha256"]) == 64 for item in protocol["candidate_encoders"][0]["artifacts"]))

    def test_protocol_freezes_required_groups_rules_and_thresholds(self):
        protocol = experiment.load_protocol(self.protocol_path)
        self.assertEqual(9, len(protocol["splits"]["outer_families"]))
        self.assertEqual(32768, protocol["input_bound"]["characters_per_prompt_or_message_content_field"])
        self.assertEqual(6, protocol["adoption_thresholds"]["positive_held_out_family_correlations_each_step"])
        self.assertEqual(0.69, protocol["adoption_thresholds"]["dev_weighted_score_minimum"])
        self.assertEqual("fail closed, do not load Dev, preserve safe-margin", protocol["runtime"]["failure_behavior"])

    def test_genuine_train_failure_does_not_load_dev(self):
        report = experiment.build_report(
            self.protocol_path,
            ROOT / "baselines/semantic-upgrade-events-evidence.v1.json",
        )
        self.assertTrue(report["feasibility"]["passed"])
        self.assertFalse(report["gates"]["dev_loaded"])
        self.assertTrue(report["gates"]["train_adoption"]["evaluated"])
        self.assertFalse(report["gates"]["train_adoption"]["passed"])
        self.assertEqual("safe-margin", report["decision"]["submission_default"])
        self.assertNotIn("data/dev", json.dumps(report))

    def test_report_regeneration_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            left = pathlib.Path(directory) / "left.json"
            right = pathlib.Path(directory) / "right.json"
            evidence = ROOT / "baselines/semantic-upgrade-events-evidence.v1.json"
            experiment.run(self.protocol_path, left, evidence)
            experiment.run(self.protocol_path, right, evidence)
            self.assertEqual(left.read_bytes(), right.read_bytes())

    def test_provisioning_downloads_only_registry_files_and_verifies(self):
        payloads = {"one.bin": b"semantic", "nested/two.bin": b"encoder"}
        protocol = json.loads(self.protocol_path.read_text())
        protocol["candidate_encoders"][0]["artifacts"] = [
            {"path": name, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in payloads.items()
        ]
        protocol["candidate_encoders"][0]["build_time_download"]["url_template"] = "https://invalid.example/{artifact_path}"
        opened = []
        class Response(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *args): self.close()
        def opener(url, timeout):
            opened.append((url, timeout))
            return Response(payloads[url.rsplit("/", 1)[-1]] if "nested/" not in url else payloads["nested/two.bin"])
        with tempfile.TemporaryDirectory() as directory:
            result = experiment.provision_artifacts(protocol, pathlib.Path(directory), opener=opener)
            self.assertTrue(result["passed"])
            self.assertEqual(2, len(opened))
            experiment.provision_artifacts(protocol, pathlib.Path(directory), opener=opener)
            self.assertEqual(2, len(opened), "verified cache must not redownload")


class EventTargetTest(unittest.TestCase):
    def test_three_way_event_labels(self):
        labels = events.event_labels([0.5, 0.5, 0.5], [0.25, 0.5, 0.75])
        self.assertEqual([0, 1, 2], labels.tolist())

    def test_expected_utility_penalizes_losses_and_ignores_tie_magnitude(self):
        probabilities = np.asarray([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
        result = events.expected_signed_utility(probabilities, 0.4, 0.25)
        np.testing.assert_allclose(result, [0.15, -0.16])

    def test_conditional_magnitudes_remain_separate(self):
        gains = np.asarray([-0.4, 0.0, 0.2, 0.6])
        labels = events.event_labels(np.zeros(4), gains)
        win, loss = events.conditional_magnitudes(gains, labels)
        self.assertAlmostEqual(0.4, win)
        self.assertAlmostEqual(0.4, loss)

    def test_inverse_frequency_balances_each_present_class(self):
        labels = np.asarray([0, 0, 0, 1, 2, 2])
        weights = events.normalized_inverse_frequency_weights(labels)
        totals = [weights[labels == label].sum() for label in range(3)]
        np.testing.assert_allclose(totals, [2.0, 2.0, 2.0])
        self.assertAlmostEqual(1.0, float(weights.mean()))

    def test_fold_local_event_model_returns_probabilities_and_signed_utility(self):
        matrix = np.asarray([[1, 0], [.8, .2], [0, 1], [.2, .8], [-1, 0], [-.8, -.2]], dtype=float)
        gains = np.asarray([.4, .2, 0, 0, -.3, -.1])
        model = events.fit_event_utility_model(matrix, gains, classifier_l2=1.0, magnitude_ridge_l2=10.0)
        prediction = events.predict_event_utility(model, matrix)
        np.testing.assert_allclose(prediction["probabilities"].sum(axis=1), 1.0)
        self.assertGreater(prediction["expected_utility"][0], prediction["expected_utility"][-1])
        self.assertEqual("softmax", model["calibration"]["method"])


class SplitAndSelectionTest(unittest.TestCase):
    def test_outer_and_inner_family_splits_never_leak(self):
        families = tuple(name for name in ("a", "b", "c", "d") for _ in range(2))
        for outer in events.nested_family_splits(families):
            outer_train = {families[index] for index in outer["train"]}
            outer_test = {families[index] for index in outer["test"]}
            self.assertFalse(outer_train & outer_test)
            for inner in outer["inner"]:
                inner_train = {families[index] for index in inner["train"]}
                inner_test = {families[index] for index in inner["test"]}
                self.assertFalse(inner_train & inner_test)
                self.assertFalse(inner_test & outer_test)

    def test_steps_can_select_different_candidates(self):
        chosen = events.select_step_candidates({
            "ax31-light->ax31": {"semantic": 0.3, "semantic+structural": 0.2},
            "ax31->axk1-think": {"semantic": 0.1, "semantic+structural": 0.4},
        })
        self.assertEqual("semantic", chosen["ax31-light->ax31"])
        self.assertEqual("semantic+structural", chosen["ax31->axk1-think"])


class BoundedRuntimeFeatureTest(unittest.TestCase):
    def test_documented_query_prefix_and_roles_are_serialized(self):
        episode = Episode("ignored", messages=(Message("system", "rules"), Message("user", "question")))
        self.assertEqual("query: [system]\nrules\n[user]\nquestion", events.semantic_text(episode))

    def test_prompt_tail_is_invariant_and_episode_id_is_unused(self):
        prefix = "x" * events.MAX_FIELD_CHARACTERS
        left = Episode("secret-family-train", prompt=prefix + "alpha")
        right = Episode("different-dev-row", prompt=prefix + "omega")
        self.assertEqual(events.bounded_role_content(left), events.bounded_role_content(right))

    def test_each_message_content_tail_is_invariant(self):
        prefix = "m" * events.MAX_FIELD_CHARACTERS
        left = Episode("one", messages=(Message("system", prefix + "a"), Message("user", prefix + "b")))
        right = Episode("two", messages=(Message("system", prefix + "x"), Message("user", prefix + "y")))
        self.assertEqual(events.bounded_role_content(left), events.bounded_role_content(right))

    def test_semantic_text_tail_is_invariant_before_tokenization(self):
        prefix = "z" * events.MAX_FIELD_CHARACTERS
        left = Episode("one", prompt=prefix + "forbidden-tail-a")
        right = Episode("two", prompt=prefix + "forbidden-tail-b")
        self.assertEqual(events.semantic_text(left), events.semantic_text(right))

    def test_repeated_extraction_probe_is_behavioral(self):
        class FakeEncoder:
            def encode(self, episodes, batch_size=16):
                return np.asarray([[1.0, 0.0] for _ in episodes], dtype=np.float32)
        result = events.extraction_probe(FakeEncoder(), [Episode("one", prompt="hello")])
        self.assertTrue(result["byte_identical"])
        self.assertTrue(result["finite"])
        self.assertEqual(0.0, result["maximum_norm_error"])


class OodAbstentionTest(unittest.TestCase):
    training = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    labels = np.asarray([2, 2, 0, 0])

    def test_supported_agreeing_neighborhood_upgrades(self):
        result = events.local_support(
            [1.0, 0.0], self.training, self.labels,
            neighbors=2, minimum_similarity=0.8, minimum_label_agreement=1.0,
            classifier_confidence=0.8, minimum_classifier_confidence=0.6,
            committee_agreement=1.0, minimum_committee_agreement=0.75,
        )
        self.assertFalse(result["abstain"])
        self.assertEqual((0, 1), result["neighbor_indices"])

    def test_any_failed_ood_check_abstains(self):
        result = events.local_support(
            [0.0, 1.0], self.training, self.labels,
            neighbors=3, minimum_similarity=0.8, minimum_label_agreement=0.75,
            classifier_confidence=0.59, minimum_classifier_confidence=0.6,
            committee_agreement=0.5, minimum_committee_agreement=0.75,
        )
        self.assertTrue(result["abstain"])
        self.assertFalse(result["checks"]["classifier_confidence"])

    def test_support_threshold_uses_only_supplied_fold_training_rows(self):
        compact = events.cosine_support_quantile(self.training[:2], 0.05)
        fold_training_with_weak_support = np.vstack((self.training[:2], [0.0, 1.0]))
        expanded = events.cosine_support_quantile(fold_training_with_weak_support, 0.05)
        self.assertNotEqual(compact, expanded)


class FrozenNegativeReportTest(unittest.TestCase):
    def test_committed_report_matches_frozen_protocol_and_preserves_default(self):
        report = json.loads((ROOT / "baselines/semantic-upgrade-events-report.v1.json").read_text())
        protocol = ROOT / "configs/semantic-upgrade-events-protocol.v1.json"
        self.assertEqual(experiment.file_sha256(protocol), report["protocol"]["sha256"])
        self.assertFalse(report["decision"]["candidate_adopted"])
        self.assertFalse(report["decision"]["runtime_integration"])
        self.assertEqual("safe-margin", report["decision"]["submission_default"])
        self.assertFalse(report["gates"]["dev_loaded"])

    def test_committed_evidence_is_a_completed_quality_failure(self):
        evidence = json.loads((ROOT / "baselines/semantic-upgrade-events-evidence.v1.json").read_text())
        self.assertTrue(evidence["artifacts"]["passed"])
        self.assertTrue(evidence["extraction_benchmark"]["constraint_passed"])
        self.assertTrue(evidence["extraction_benchmark"]["byte_identical"])
        self.assertFalse(evidence["train_gate"]["passed"])
        self.assertTrue(all(row["status"] == "completed" for row in evidence["train_evaluation"].values()))


if __name__ == "__main__":
    unittest.main()
