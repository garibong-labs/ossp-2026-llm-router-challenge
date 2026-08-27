# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the frozen Fast double-consensus experiment."""

from __future__ import annotations

import hashlib
import inspect
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "baselines", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import risk_validation  # noqa: E402
import run_safe_margin_fast_double_consensus as runner  # noqa: E402
import safe_margin  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Episode,
    Message,
    load_bundled_policy,
    load_input,
)

LIGHT, AX31, THINK = MODEL_IDS


def _prediction(efficiency: float, signature: tuple[int, ...], increment=0.27):
    return safe_margin.EpisodePrediction(
        scores={LIGHT: 0.2, AX31: 0.2 + efficiency * increment, THINK: 0.8},
        costs={LIGHT: 1.0, AX31: 1.0 + increment, THINK: 3.0},
        signature=signature,
    )


def _swap_predictions():
    # x1 ties these within one octave and chooses signature (1,). x3 strictly
    # prefers signature (9,), and the residual signal agrees in the tests.
    return (
        _prediction(0.60, (1,)),
        _prediction(0.60, (1,)),
        _prediction(0.90, (9,)),
        _prediction(0.90, (9,)),
    )


class DoubleConsensusPlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()

    def test_both_signals_agree_for_exact_atomic_swap(self):
        predictions = _swap_predictions()
        baseline = safe_margin.plan_selection(predictions, self.policy, "fast")[0]
        plan = runner.matched_fast_plan(predictions, (0.01, 0.01, 0.09, 0.09), self.policy)
        self.assertEqual((AX31, AX31, LIGHT, LIGHT), baseline)
        self.assertEqual((LIGHT, LIGHT, AX31, AX31), plan.selected)
        self.assertEqual(baseline.count(AX31), plan.selected.count(AX31))
        self.assertLessEqual(plan.predicted_increment, plan.baseline_increment)
        self.assertNotIn(THINK, plan.selected)
        self.assertEqual(1, plan.swaps)
        self.assertFalse(plan.fallback_used)

    def test_residual_disagreement_or_tie_retains_x1(self):
        predictions = _swap_predictions()
        baseline = safe_margin.plan_selection(predictions, self.policy, "fast")[0]
        for residuals in (
            (0.09, 0.09, 0.01, 0.01),
            (0.05, 0.05, 0.05, 0.05),
        ):
            with self.subTest(residuals=residuals):
                plan = runner.matched_fast_plan(predictions, residuals, self.policy)
                self.assertEqual(baseline, plan.selected)
                self.assertEqual(0, plan.swaps)

    def test_efficiency_tie_retains_x1_even_when_residual_prefers_incoming(self):
        predictions = tuple(_prediction(0.60, signature) for signature in ((1,), (1,), (9,), (9,)))
        baseline = safe_margin.plan_selection(predictions, self.policy, "fast")[0]
        plan = runner.matched_fast_plan(predictions, (0.01, 0.01, 0.09, 0.09), self.policy)
        self.assertEqual(baseline, plan.selected)
        self.assertEqual(0, plan.swaps)

    def test_missing_malformed_nonfinite_signal_falls_back_to_x1(self):
        predictions = _swap_predictions()
        baseline = safe_margin.plan_selection(predictions, self.policy, "fast")[0]
        for residuals in ((), (0.0, 0.0, float("nan"), 0.0), (True, 0.0, 0.0, 0.0)):
            with self.subTest(residuals=residuals):
                plan = runner.matched_fast_plan(predictions, residuals, self.policy)
                self.assertEqual(baseline, plan.selected)
                self.assertTrue(plan.fallback_used)

    def test_costlier_or_wrong_cardinality_candidate_retains_x1(self):
        predictions = list(_swap_predictions())
        predictions[2] = _prediction(0.90, (9,), increment=0.29)
        baseline = safe_margin.plan_selection(tuple(predictions), self.policy, "fast")[0]
        plan = runner.matched_fast_plan(
            tuple(predictions), (0.01, 0.01, 0.09, 0.09), self.policy
        )
        self.assertEqual(baseline, plan.selected)

    def test_content_signature_order_is_repeatable_and_order_invariant(self):
        predictions = _swap_predictions()
        residuals = (0.01, 0.01, 0.09, 0.09)
        first = runner.matched_fast_plan(predictions, residuals, self.policy)
        second = runner.matched_fast_plan(predictions, residuals, self.policy)
        backward = runner.matched_fast_plan(
            tuple(reversed(predictions)), tuple(reversed(residuals)), self.policy
        )
        self.assertEqual(first, second)
        self.assertEqual(first.selected, tuple(reversed(backward.selected)))

    def test_balanced_and_premium_are_exact_x1(self):
        inputs = load_input(ROOT / "data/toy/inputs.json")
        artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
        predictions = safe_margin.predict_batch(inputs.episodes, artifact, self.policy)
        residuals = (0.0,) * len(inputs.episodes)
        for tier in ("balanced", "premium"):
            baseline = safe_margin.make_safe_margin_submission(
                inputs, self.policy, artifact, tier
            ).submission
            candidate, plan = runner.candidate_submission(
                inputs, self.policy, artifact, tier, predictions, residuals
            )
            self.assertEqual(baseline, candidate)
            self.assertIsNone(plan)


class RuntimeIsolationTest(unittest.TestCase):
    def test_residual_is_episode_id_independent(self):
        artifact = runner.load_residual_artifact()
        prompt_a = Episode("train-family-secret", prompt="Prove that 2 + 2 = 4.")
        prompt_b = Episode("dev-outcome-secret", prompt="Prove that 2 + 2 = 4.")
        self.assertEqual(
            runner.predict_residual(prompt_a, artifact),
            runner.predict_residual(prompt_b, artifact),
        )
        messages_a = Episode("one", messages=(Message("user", "질문입니다"),))
        messages_b = Episode("two", messages=(Message("user", "질문입니다"),))
        self.assertEqual(
            runner.predict_residual(messages_a, artifact),
            runner.predict_residual(messages_b, artifact),
        )

    def test_runtime_selection_functions_have_no_family_or_outcome_input(self):
        for function in (
            runner.predict_residual,
            runner.predict_residuals,
            runner._eligible_groups,
            runner.matched_fast_plan,
            runner.candidate_submission,
        ):
            parameters = set(inspect.signature(function).parameters)
            self.assertFalse(parameters & {"family", "families", "outcome", "outcomes"})

    def test_production_default_files_are_byte_unchanged(self):
        self.assertEqual(1, safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE)
        expected = {
            ROOT / "baselines/safe_margin.py": "49ffb62abd0cefc04e248a012074f823a725ad86afad963aaf6d1261004b8402",
            ROOT / "container/entrypoint.py": "04ccd502db35717c6a1b342d84f516916285f4a9061032a9bd727bbb5488527b",
        }
        for path, digest in expected.items():
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())


def _family_values(positive=4, negative=None):
    values = {
        family: ("0.0001" if index < positive else "0")
        for index, family in enumerate(risk_validation.FAMILY_LABELS)
    }
    if negative is not None:
        values[risk_validation.FAMILY_LABELS[-1]] = negative
    return values


def _train_evidence(**overrides):
    value = {
        "weighted_score_delta": "0.0001",
        "fast_quality_delta": "0.0001",
        "family_deltas": _family_values(),
        "balanced_premium_exact_x1": True,
        "fast_model_counts_exact_x1": True,
        "candidate_fast_predicted_spend_lte_x1": True,
        "deterministic_repeated_output": True,
    }
    value.update(overrides)
    return value


class GateBoundaryTest(unittest.TestCase):
    def test_complete_train_boundary_passes(self):
        gate = runner.decide_train_gate(_train_evidence())
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(4, gate["positive_families"])
        self.assertEqual(9, gate["nonnegative_families"])

    def test_strict_train_boundaries_and_all_invariants(self):
        self.assertFalse(runner.decide_train_gate(_train_evidence(weighted_score_delta="0"))["gate_passed"])
        self.assertFalse(runner.decide_train_gate(_train_evidence(fast_quality_delta="0"))["gate_passed"])
        self.assertFalse(runner.decide_train_gate(_train_evidence(family_deltas=_family_values(negative="-0.000000000001")))["gate_passed"])
        self.assertFalse(runner.decide_train_gate(_train_evidence(family_deltas=_family_values(positive=3)))["gate_passed"])
        for name in (
            "balanced_premium_exact_x1", "fast_model_counts_exact_x1",
            "candidate_fast_predicted_spend_lte_x1", "deterministic_repeated_output",
        ):
            self.assertFalse(runner.decide_train_gate(_train_evidence(**{name: False}))["gate_passed"])

    def test_dev_boundary_is_strict_and_fail_closed(self):
        evidence = {
            "candidate_score": "0.673182",
            "balanced_premium_exact_x1": True,
            "fast_model_counts_exact_x1": True,
            "candidate_fast_predicted_spend_lte_x1": True,
        }
        self.assertFalse(runner.decide_dev_gate(evidence)["gate_passed"])
        evidence["candidate_score"] = "0.673182000001"
        self.assertTrue(runner.decide_dev_gate(evidence)["gate_passed"])
        self.assertTrue(runner.decide_train_gate({})["malformed"])
        for value in ("nan", "inf", None, True):
            self.assertFalse(runner.decide_train_gate(_train_evidence(weighted_score_delta=value))["gate_passed"])


def _safety_policy(p99=1.1, maximum=1.2, breaches=0, whole=True, holdout=True):
    return {
        "tiers": {
            tier: {
                "whole_split": {"budget_passed": whole},
                "bootstrap": {"cap_breaches": breaches, "cost_ratio": {"p99": p99, "max": maximum}},
                "family_holdouts": {
                    family: {"evaluated": True, "budget_passed": holdout}
                    for family in risk_validation.FAMILY_LABELS
                },
            }
            for tier in TIERS
        }
    }


class SafetyGateTest(unittest.TestCase):
    def test_exact_contract_and_fail_closed_boundaries(self):
        reference = _safety_policy()
        self.assertTrue(runner.decide_safety_gate(reference, reference)["gate_passed"])
        for candidate in (
            _safety_policy(p99=1.1000001), _safety_policy(maximum=1.2000001),
            _safety_policy(breaches=1), _safety_policy(whole=False),
            _safety_policy(holdout=False),
        ):
            self.assertFalse(runner.decide_safety_gate(reference, candidate)["gate_passed"])
        self.assertTrue(runner.decide_safety_gate({}, {})["malformed"])


class AccessAndReportTest(unittest.TestCase):
    def test_train_failure_never_opens_dev(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "evaluate_train", return_value={"gate_passed": False}
        ), mock.patch.object(
            runner, "_load_split_once", side_effect=AssertionError("Dev opened")
        ):
            report = runner.run_experiment(report_path=pathlib.Path(directory) / "report.json")
        self.assertEqual(0, report["dev"]["load_count"])
        self.assertEqual(0, report["dev"]["evaluation_count"])
        self.assertEqual(0, report["safety"]["evaluation_count"])

    def test_dev_loads_once_and_reuses_loaded_data_for_safety(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "evaluate_train", return_value={"gate_passed": True}
        ), mock.patch.object(
            runner, "_load_split_once", return_value=sentinel
        ) as loader, mock.patch.object(
            runner, "evaluate_dev_loaded",
            return_value={"accessed": True, "load_count": 1, "evaluation_count": 1, "gate_passed": True},
        ) as dev, mock.patch.object(
            runner, "evaluate_safety_loaded",
            return_value={"accessed": True, "evaluation_count": 1, "gate_passed": True},
        ) as safety:
            report = runner.run_experiment(report_path=pathlib.Path(directory) / "report.json")
        loader.assert_called_once()
        dev.assert_called_once_with(sentinel)
        safety.assert_called_once_with(sentinel)
        self.assertTrue(report["decision"]["eligible"])

    def test_dev_failure_never_runs_safety(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "evaluate_train", return_value={"gate_passed": True}
        ), mock.patch.object(
            runner, "_load_split_once", return_value=object()
        ), mock.patch.object(
            runner, "evaluate_dev_loaded",
            return_value={"accessed": True, "load_count": 1, "evaluation_count": 1, "gate_passed": False},
        ), mock.patch.object(
            runner, "evaluate_safety_loaded", side_effect=AssertionError("safety ran")
        ):
            report = runner.run_experiment(report_path=pathlib.Path(directory) / "report.json")
        self.assertEqual(1, report["dev"]["evaluation_count"])
        self.assertEqual(0, report["safety"]["evaluation_count"])

    def test_frozen_hashes_and_byte_stable_reemit(self):
        self.assertEqual(runner.EXPECTED_PROTOCOL_SHA256, runner.file_sha256(runner.PROTOCOL_PATH))
        self.assertEqual(runner.EXPECTED_ARTIFACT_SHA256, runner.file_sha256(runner.ARTIFACT_PATH))
        runner.verify_protocol()
        value = {
            "report_type": runner.REPORT_TYPE,
            "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
            "artifact_sha256": runner.EXPECTED_ARTIFACT_SHA256,
            "experiment_id": runner.EXPERIMENT_ID,
            "decision": {"submission_default": "safe-margin"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            runner._atomic_json(path, value)
            first = path.read_bytes()
            with mock.patch.object(runner, "evaluate_train", side_effect=AssertionError("Train opened")), mock.patch.object(runner, "_load_split_once", side_effect=AssertionError("Dev opened")):
                runner.reemit_existing_report(path)
            self.assertEqual(first, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
