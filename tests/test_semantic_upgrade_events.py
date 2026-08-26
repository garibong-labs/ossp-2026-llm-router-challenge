# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
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
EVIDENCE_PATH = ROOT / "baselines/semantic-upgrade-events-evidence.v1.json"
REPORT_PATH = ROOT / "baselines/semantic-upgrade-events-report.v1.json"
TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"
TRAIN_OUTCOMES = ROOT / "data/train/outcomes.json"
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


def _measurement_record(evidence, key="official_linux_arm64_feasibility"):
    """Rebuild the extraction measurement record embedded in the evidence."""

    block = evidence[key]
    return {
        "record_type": experiment.RECORD_TYPE,
        "protocol_sha256": evidence["protocol_sha256"],
        "train_input_sha256": evidence["train_input_sha256"],
        "train_outcomes_sha256": evidence["train_outcomes_sha256"],
        "artifacts": block["artifacts"],
        "dependencies": block["dependencies"],
        "benchmark": block["extraction_benchmark"],
    }


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

    def test_native_preflight_is_separate_and_never_opens_the_official_gate(self):
        report = experiment.build_report(self.protocol_path, EVIDENCE_PATH)
        preflight = report["feasibility"]["native_apple_arm64_preflight"]
        self.assertEqual("darwin/arm64", preflight["required_environment"])
        self.assertTrue(preflight["evaluated"])
        self.assertTrue(preflight["passed"])
        self.assertFalse(preflight["qualifies_official_feasibility"])
        environment = preflight["extraction_benchmark"]["environment"]
        self.assertTrue(environment["native_apple_arm64"])
        self.assertFalse(environment["official_architecture_match"])
        self.assertFalse(preflight["extraction_benchmark"]["enforced_limits"]["all_enforced"])
        official = report["feasibility"]["official_linux_arm64"]
        self.assertIsNot(preflight["extraction_benchmark"], official["extraction_benchmark"])
        self.assertNotEqual(
            environment["label"], official["extraction_benchmark"]["environment"]["label"]
        )

    def test_official_gate_needs_official_architecture_and_enforced_limits(self):
        protocol = experiment.load_protocol(self.protocol_path)
        evidence = json.loads(EVIDENCE_PATH.read_text())
        record = _measurement_record(evidence)
        self.assertTrue(experiment.official_feasibility_evidence(protocol, record)["passed"])
        for mutate, status in (
            (lambda row: row["benchmark"]["environment"].update(machine="x86_64", official_architecture_match=False), "not-official-architecture"),
            (lambda row: row["benchmark"]["environment"].update(containerized=False), "official-architecture-without-container-isolation"),
            (lambda row: row["benchmark"]["enforced_limits"].update(all_enforced=False), "official-architecture-without-enforced-frozen-limits"),
        ):
            with self.subTest(status=status):
                broken = copy.deepcopy(record)
                mutate(broken)
                result = experiment.official_feasibility_evidence(protocol, broken)
                self.assertFalse(result["evaluated"])
                self.assertFalse(result["passed"])
                self.assertEqual(status, result["status"])
        exceeded = copy.deepcopy(record)
        exceeded["benchmark"]["constraint_passed"] = False
        result = experiment.official_feasibility_evidence(protocol, exceeded)
        self.assertTrue(result["evaluated"])
        self.assertFalse(result["passed"])

    def test_enforced_limits_are_compared_against_the_frozen_protocol(self):
        limits = experiment.frozen_limits(experiment.load_protocol(self.protocol_path))
        environment = {
            "network_isolated": True,
            "cgroup": {"values": {
                "cpu.max": "200000 100000", "memory.max": "2147483648",
                "memory.swap.max": "0", "pids.max": "32",
            }},
        }
        self.assertTrue(experiment.enforced_limit_evidence(environment, limits)["all_enforced"])
        for key, value, check in (
            ("cpu.max", "400000 100000", "cpu_quota"),
            ("memory.max", "4294967296", "memory_max"),
            ("memory.swap.max", "max", "no_additional_swap"),
            ("pids.max", "64", "pids_max"),
        ):
            with self.subTest(check=check):
                loosened = copy.deepcopy(environment)
                loosened["cgroup"]["values"][key] = value
                result = experiment.enforced_limit_evidence(loosened, limits)
                self.assertFalse(result["checks"][check])
                self.assertFalse(result["all_enforced"])
        offline = copy.deepcopy(environment)
        offline["network_isolated"] = False
        self.assertFalse(experiment.enforced_limit_evidence(offline, limits)["all_enforced"])

    def test_measurement_image_stays_out_of_the_submission_path(self):
        measurement = (ROOT / "container/semantic-measurement.Dockerfile").read_text()
        self.assertIn('io.sktelecom.ossp.submission-image="false"', measurement)
        self.assertIn("--verify-only --encoder-dir /opt/encoder", measurement)
        self.assertNotIn("safe_margin", measurement)
        submission = (ROOT / "container/Dockerfile").read_text()
        self.assertIn("baselines/safe_margin.py", submission)
        for forbidden in ("encoder", "onnx", "semantic"):
            with self.subTest(token=forbidden):
                self.assertNotIn(forbidden, submission)
                self.assertNotIn(forbidden, (ROOT / ".dockerignore").read_text())

    @unittest.skipUnless(TRAIN_INPUT.is_file(), "public Train materialization is required")
    def test_evidence_rebuild_refuses_unverified_measurement_inputs(self):
        evidence = json.loads(EVIDENCE_PATH.read_text())
        record = _measurement_record(evidence)
        benchmark = record["benchmark"]
        foreign = np.zeros((benchmark["rows"], benchmark["dimensions"]), dtype=np.float32)
        with self.subTest(reason="foreign embedding matrix"):
            with self.assertRaises(ValueError):
                experiment.build_evidence(
                    self.protocol_path, record, foreign, TRAIN_INPUT, TRAIN_OUTCOMES
                )
        for field, value in (("byte_identical", False), ("maximum_norm_error", 1.0)):
            with self.subTest(reason=field):
                broken = copy.deepcopy(record)
                broken["benchmark"][field] = value
                with self.assertRaises(ValueError):
                    experiment.build_evidence(
                        self.protocol_path, broken, foreign, TRAIN_INPUT, TRAIN_OUTCOMES
                    )
        with self.subTest(reason="protocol drift"):
            drifted = copy.deepcopy(record)
            drifted["protocol_sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                experiment.build_evidence(
                    self.protocol_path, drifted, foreign, TRAIN_INPUT, TRAIN_OUTCOMES
                )

    def test_report_regeneration_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            left = pathlib.Path(directory) / "left.json"
            right = pathlib.Path(directory) / "right.json"
            experiment.run(self.protocol_path, left, EVIDENCE_PATH)
            experiment.run(self.protocol_path, right, EVIDENCE_PATH)
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
        report = json.loads(REPORT_PATH.read_text())
        protocol = ROOT / "configs/semantic-upgrade-events-protocol.v1.json"
        self.assertEqual(experiment.file_sha256(protocol), report["protocol"]["sha256"])
        self.assertFalse(report["decision"]["candidate_adopted"])
        self.assertFalse(report["decision"]["runtime_integration"])
        self.assertEqual("safe-margin", report["decision"]["submission_default"])
        self.assertTrue(report["feasibility"]["passed"])
        self.assertEqual("linux/arm64", report["gates"]["feasibility"]["required_environment"])
        self.assertTrue(report["gates"]["feasibility"]["evaluated"])
        self.assertTrue(report["gates"]["feasibility"]["passed"])
        self.assertFalse(report["gates"]["feasibility"]["final_contest_hardware"])
        self.assertFalse(report["gates"]["dev_loaded"])
        self.assertTrue(report["gates"]["train_adoption"]["evaluated"])
        self.assertFalse(report["gates"]["train_adoption"]["passed"])
        for closed in ("dev_champion", "calibration_conformal", "safety_5000_resamples", "official_container_benchmark"):
            with self.subTest(gate=closed):
                self.assertFalse(report["gates"][closed]["evaluated"])
                self.assertFalse(report["gates"][closed]["passed"])
        self.assertNotIn("data/dev", json.dumps(report))

    def test_committed_report_never_claims_the_final_contest_device(self):
        report = json.loads(REPORT_PATH.read_text())
        limitations = " ".join(report["limitations"])
        self.assertIn("not the operator's final contest device", limitations)
        self.assertIn("Native Apple arm64 preflight validates extraction only", limitations)
        self.assertFalse(report["feasibility"]["official_linux_arm64"]["final_contest_hardware"])

    def test_committed_evidence_measured_the_official_architecture_under_frozen_limits(self):
        evidence = json.loads(EVIDENCE_PATH.read_text())
        official = evidence["official_linux_arm64_feasibility"]
        self.assertTrue(official["evaluated"])
        self.assertTrue(official["passed"])
        self.assertEqual("measured-on-local-linux-arm64-container", official["status"])
        self.assertFalse(official["final_contest_hardware"])
        benchmark = official["extraction_benchmark"]
        environment = benchmark["environment"]
        self.assertEqual("Linux", environment["system"])
        self.assertIn(environment["machine"], ("aarch64", "arm64"))
        self.assertTrue(environment["containerized"])
        self.assertTrue(environment["network_isolated"])
        self.assertTrue(environment["root_filesystem_read_only"])
        self.assertTrue(environment["cgroup"]["unified_v2"])
        self.assertTrue(benchmark["enforced_limits"]["all_enforced"])
        limits = benchmark["limits"]
        self.assertEqual(1760, benchmark["rows"])
        self.assertLessEqual(benchmark["first_seconds"], limits["seconds_per_tier"])
        self.assertLessEqual(benchmark["second_seconds"], limits["seconds_per_tier"])
        self.assertLessEqual(benchmark["observed_peak_memory_bytes"], limits["memory_max_bytes"])
        self.assertLessEqual(benchmark["maximum_pid_thread_observation"], limits["pid_thread_limit"])
        self.assertIsNotNone(benchmark["cgroup_memory_peak_bytes"])
        self.assertIsNotNone(benchmark["cgroup_pids_peak"])
        self.assertTrue(benchmark["byte_identical"])
        self.assertEqual(benchmark["first_sha256"], benchmark["second_sha256"])

    def test_committed_train_evaluation_came_from_the_linux_arm64_embeddings(self):
        evidence = json.loads(EVIDENCE_PATH.read_text())
        source = evidence["train_embedding_source"]
        benchmark = evidence["official_linux_arm64_feasibility"]["extraction_benchmark"]
        self.assertTrue(source["official_architecture_match"])
        self.assertEqual(benchmark["environment"]["label"], source["environment_label"])
        self.assertEqual(benchmark["first_sha256"], source["first_pass_sha256"])
        self.assertEqual(benchmark["second_sha256"], source["second_pass_sha256"])
        self.assertTrue(source["byte_identical"])

    def test_committed_evidence_is_a_completed_quality_failure(self):
        evidence = json.loads(EVIDENCE_PATH.read_text())
        self.assertTrue(evidence["artifacts"]["passed"])
        preflight = evidence["native_apple_arm64_preflight"]
        self.assertTrue(preflight["evaluated"])
        self.assertTrue(preflight["passed"])
        self.assertFalse(preflight["qualifies_official_feasibility"])
        self.assertTrue(preflight["extraction_benchmark"]["constraint_passed"])
        self.assertTrue(preflight["extraction_benchmark"]["byte_identical"])
        self.assertFalse(evidence["train_gate"]["passed"])
        self.assertTrue(evidence["train_gate"]["evaluated"])
        self.assertTrue(all(row["status"] == "completed" for row in evidence["train_evaluation"].values()))


if __name__ == "__main__":
    unittest.main()
