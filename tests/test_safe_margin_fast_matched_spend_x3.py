# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the frozen Fast matched-spend x3 experiment."""

from __future__ import annotations

import hashlib
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
import run_safe_margin_fast_matched_spend_x3 as runner  # noqa: E402
import safe_margin  # noqa: E402
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy  # noqa: E402

LIGHT, AX31, THINK = MODEL_IDS


def _prediction(efficiency: float, signature: tuple[int, ...]):
    increment = 0.27
    return safe_margin.EpisodePrediction(
        scores={LIGHT: 0.2, AX31: 0.2 + efficiency * increment, THINK: 0.8},
        costs={LIGHT: 1.0, AX31: 1.0 + increment, THINK: 3.0},
        signature=signature,
    )


def _swap_predictions():
    # x1 puts both signatures in the same octave and chooses signature (1,).
    # x3 sees signature (9,) as strictly more efficient and swaps the groups.
    return (
        _prediction(0.60, (1,)),
        _prediction(0.60, (1,)),
        _prediction(0.90, (9,)),
        _prediction(0.90, (9,)),
    )


class MatchedPlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()

    def test_default_x1_path_is_unchanged(self):
        self.assertEqual(1, safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE)
        predictions = _swap_predictions()
        first = safe_margin.plan_selection(predictions, self.policy, "fast")
        second = safe_margin.plan_selection(predictions, self.policy, "fast")
        self.assertEqual(first, second)

    def test_only_fast_changes_and_think_is_never_introduced(self):
        plan = runner.matched_fast_plan(_swap_predictions(), self.policy)
        baseline = safe_margin.plan_selection(_swap_predictions(), self.policy, "fast")[0]
        self.assertNotEqual(baseline, plan.selected)
        self.assertNotIn(THINK, plan.selected)

    def test_exact_count_spend_and_group_atomicity(self):
        predictions = _swap_predictions()
        baseline = safe_margin.plan_selection(predictions, self.policy, "fast")[0]
        plan = runner.matched_fast_plan(predictions, self.policy)
        self.assertEqual(baseline.count(AX31), plan.selected.count(AX31))
        self.assertLessEqual(plan.predicted_increment, plan.baseline_increment)
        self.assertEqual(plan.selected[0], plan.selected[1])
        self.assertEqual(plan.selected[2], plan.selected[3])
        self.assertEqual((LIGHT, LIGHT, AX31, AX31), plan.selected)
        self.assertEqual(1, plan.swaps)

    def test_ordering_is_content_deterministic(self):
        forward = runner.matched_fast_plan(_swap_predictions(), self.policy)
        backward = runner.matched_fast_plan(tuple(reversed(_swap_predictions())), self.policy)
        self.assertEqual(forward.selected, tuple(reversed(backward.selected)))
        self.assertEqual(forward.predicted_ratio, backward.predicted_ratio)

    def test_costlier_or_wrong_cardinality_incoming_group_keeps_x1(self):
        predictions = list(_swap_predictions())
        costly = predictions[2]
        predictions[2] = safe_margin.EpisodePrediction(
            scores=costly.scores,
            costs={LIGHT: 1.0, AX31: 1.29, THINK: 3.0},
            signature=costly.signature,
        )
        # The high-signature group no longer has equal per-group spend.
        plan = runner.matched_fast_plan(tuple(predictions), self.policy)
        baseline = safe_margin.plan_selection(tuple(predictions), self.policy, "fast")[0]
        self.assertEqual(baseline, plan.selected)


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

    def test_train_deltas_are_strict_and_worst_family_is_zero(self):
        self.assertFalse(runner.decide_train_gate(_train_evidence(weighted_score_delta="0"))["gate_passed"])
        self.assertFalse(runner.decide_train_gate(_train_evidence(fast_quality_delta="0"))["gate_passed"])
        self.assertFalse(runner.decide_train_gate(_train_evidence(family_deltas=_family_values(negative="-0.000000000001")))["gate_passed"])

    def test_every_invariant_is_required(self):
        for name in (
            "balanced_premium_exact_x1",
            "fast_model_counts_exact_x1",
            "candidate_fast_predicted_spend_lte_x1",
            "deterministic_repeated_output",
        ):
            with self.subTest(name=name):
                self.assertFalse(runner.decide_train_gate(_train_evidence(**{name: False}))["gate_passed"])

    def test_dev_boundary_is_strict_and_requires_invariants(self):
        evidence = {
            "candidate_score": "0.673182",
            "balanced_premium_exact_x1": True,
            "fast_model_counts_exact_x1": True,
            "candidate_fast_predicted_spend_lte_x1": True,
        }
        self.assertFalse(runner.decide_dev_gate(evidence)["gate_passed"])
        evidence["candidate_score"] = "0.673182000001"
        self.assertTrue(runner.decide_dev_gate(evidence)["gate_passed"])
        evidence["fast_model_counts_exact_x1"] = False
        self.assertFalse(runner.decide_dev_gate(evidence)["gate_passed"])

    def test_missing_malformed_and_nonfinite_evidence_fails_closed(self):
        self.assertTrue(runner.decide_train_gate({})["malformed"])
        for value in ("nan", "inf", None, True):
            evidence = _train_evidence(weighted_score_delta=value)
            self.assertFalse(runner.decide_train_gate(evidence)["gate_passed"])
        self.assertTrue(runner.decide_dev_gate({})["malformed"])


def _safety_policy(p99=1.1, maximum=1.2, breaches=0, whole=True, holdout=True):
    return {
        "tiers": {
            tier: {
                "whole_split": {"budget_passed": whole},
                "bootstrap": {"cap_breaches": breaches, "cost_ratio": {"p99": p99, "max": maximum}},
                "family_holdouts": {family: {"evaluated": True, "budget_passed": holdout} for family in risk_validation.FAMILY_LABELS},
            }
            for tier in TIERS
        }
    }


class SafetyGateTest(unittest.TestCase):
    def test_matching_tail_and_zero_breaches_pass(self):
        report = _safety_policy()
        self.assertTrue(runner.decide_safety_gate(report, report)["gate_passed"])

    def test_worse_tail_or_any_breach_fails(self):
        reference = _safety_policy()
        for candidate in (_safety_policy(p99=1.1000001), _safety_policy(maximum=1.2000001), _safety_policy(breaches=1), _safety_policy(whole=False), _safety_policy(holdout=False)):
            self.assertFalse(runner.decide_safety_gate(reference, candidate)["gate_passed"])
        self.assertTrue(runner.decide_safety_gate({}, {})["malformed"])


class AccessAndReportTest(unittest.TestCase):
    def test_train_failure_never_opens_dev(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "evaluate_train", return_value={"gate_passed": False}), mock.patch.object(runner, "_load_split_once", side_effect=AssertionError("Dev opened")):
            report = runner.run_experiment(report_path=pathlib.Path(directory) / "report.json")
        self.assertEqual(0, report["dev"]["evaluation_count"])
        self.assertFalse(report["dev"]["accessed"])

    def test_dev_is_loaded_once_and_reused_for_safety(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "evaluate_train", return_value={"gate_passed": True}), mock.patch.object(runner, "_load_split_once", return_value=sentinel) as loader, mock.patch.object(runner, "evaluate_dev_loaded", return_value={"accessed": True, "evaluation_count": 1, "gate_passed": True}) as dev, mock.patch.object(runner, "evaluate_safety_loaded", return_value={"accessed": True, "evaluation_count": 1, "gate_passed": True}) as safety:
            report = runner.run_experiment(report_path=pathlib.Path(directory) / "report.json")
        loader.assert_called_once()
        dev.assert_called_once_with(sentinel)
        safety.assert_called_once_with(sentinel)
        self.assertTrue(report["decision"]["eligible"])

    def test_dev_failure_never_runs_safety(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(runner, "evaluate_train", return_value={"gate_passed": True}), mock.patch.object(runner, "_load_split_once", return_value=object()), mock.patch.object(runner, "evaluate_dev_loaded", return_value={"accessed": True, "evaluation_count": 1, "gate_passed": False}), mock.patch.object(runner, "evaluate_safety_loaded", side_effect=AssertionError("safety ran")):
            report = runner.run_experiment(report_path=pathlib.Path(directory) / "report.json")
        self.assertEqual(0, report["safety"]["evaluation_count"])

    def test_protocol_and_report_bytes_are_stable(self):
        self.assertEqual(runner.EXPECTED_PROTOCOL_SHA256, hashlib.sha256(runner.PROTOCOL_PATH.read_bytes()).hexdigest())
        runner.verify_protocol()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            value = {"z": 3, "a": {"eligible": False}}
            runner._atomic_json(path, value)
            first = path.read_bytes()
            runner._atomic_json(path, value)
            self.assertEqual(first, path.read_bytes())

    def test_reemit_does_not_access_train_or_dev(self):
        value = {"report_type": runner.REPORT_TYPE, "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256, "experiment_id": runner.EXPERIMENT_ID, "decision": {"submission_default": "safe-margin"}}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            runner._atomic_json(path, value)
            first = path.read_bytes()
            with mock.patch.object(runner, "evaluate_train", side_effect=AssertionError("Train opened")), mock.patch.object(runner, "_load_split_once", side_effect=AssertionError("Dev opened")):
                runner.reemit_existing_report(path)
            self.assertEqual(first, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
