# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the frozen efficiency-bucket x3 experiment."""

from __future__ import annotations

import dataclasses
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
import run_safe_margin_efficiency_bucket_x3 as runner  # noqa: E402
import safe_margin  # noqa: E402
from ossp_router.protocol import MODEL_IDS, TIERS, load_bundled_policy  # noqa: E402


LIGHT, AX31, THINK = MODEL_IDS


def _prediction(efficiency: float, signature=(1,)):
    increment = 0.19
    return safe_margin.EpisodePrediction(
        scores={LIGHT: 0.2, AX31: 0.2 + efficiency * increment, THINK: 0.8},
        costs={LIGHT: 1.0, AX31: 1.0 + increment, THINK: 3.0},
        signature=tuple(signature),
    )


class PlannerParameterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_bundled_policy()
        cls.config = dataclasses.replace(
            safe_margin.TIER_PLAN_CONFIGS["balanced"],
            target_ratio=1.095,
            ax31_min_gain=0.0,
            ax31_max_step_ratio=100.0,
            ax31_max_step_load=100.0,
        )

    def test_x3_has_exactly_three_log_buckets_per_octave(self):
        self.assertEqual(0, safe_margin._efficiency_bucket(1.0, 3))
        self.assertEqual(1, safe_margin._efficiency_bucket(1.3, 3))
        self.assertEqual(2, safe_margin._efficiency_bucket(1.7, 3))
        self.assertEqual(3, safe_margin._efficiency_bucket(2.0, 3))
        with self.assertRaises(ValueError):
            safe_margin._efficiency_bucket(1.0, 0)

    def test_default_is_identical_to_explicit_x1(self):
        predictions = (_prediction(0.80), _prediction(0.65))
        implicit = safe_margin.plan_selection(
            predictions, self.policy, "balanced", self.config
        )
        explicit = safe_margin.plan_selection(
            predictions,
            self.policy,
            "balanced",
            self.config,
            efficiency_buckets_per_octave=1,
        )
        self.assertEqual(implicit, explicit)
        self.assertEqual(1, safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE)

    def test_x3_refines_an_x1_group_without_changing_other_semantics(self):
        predictions = (_prediction(0.80), _prediction(0.65))
        x1 = safe_margin.plan_selection(
            predictions, self.policy, "balanced", self.config
        )
        x3 = safe_margin.plan_selection(
            predictions,
            self.policy,
            "balanced",
            self.config,
            efficiency_buckets_per_octave=3,
        )
        self.assertEqual((LIGHT, LIGHT), x1[0])
        self.assertEqual((AX31, LIGHT), x3[0])

    def test_x3_is_order_invariant_and_keeps_groups_atomic(self):
        predictions = (
            _prediction(0.80, (1,)),
            _prediction(0.80, (1,)),
            _prediction(0.65, (2,)),
        )
        config = dataclasses.replace(self.config, target_ratio=1.07)
        forward = safe_margin.plan_selection(
            predictions,
            self.policy,
            "balanced",
            config,
            efficiency_buckets_per_octave=3,
        )
        backward = safe_margin.plan_selection(
            tuple(reversed(predictions)),
            self.policy,
            "balanced",
            config,
            efficiency_buckets_per_octave=3,
        )
        self.assertEqual(list(forward[0]), list(reversed(backward[0])))
        self.assertEqual(forward[1], backward[1])
        self.assertEqual(forward[0][0], forward[0][1])


def _family_values(positive=4, negative=None):
    values = {
        family: ("0.0001" if index < positive else "0")
        for index, family in enumerate(risk_validation.FAMILY_LABELS)
    }
    if negative is not None:
        values[risk_validation.FAMILY_LABELS[-1]] = negative
    return values


class GateBoundaryTest(unittest.TestCase):
    def test_exact_train_pass_boundary(self):
        gate = runner.decide_train_gate("0.000000000001", _family_values())
        self.assertTrue(gate["gate_passed"])
        self.assertEqual(4, gate["positive_families"])
        self.assertEqual(9, gate["nonnegative_families"])

    def test_weighted_delta_is_strict(self):
        self.assertFalse(
            runner.decide_train_gate("0", _family_values())["gate_passed"]
        )

    def test_positive_and_nonnegative_family_boundaries_are_exact(self):
        self.assertFalse(
            runner.decide_train_gate("0.1", _family_values(positive=3))[
                "checks"
            ]["positive_families"]
        )
        gate = runner.decide_train_gate(
            "0.1", _family_values(negative="-0.0005")
        )
        self.assertTrue(gate["checks"]["worst_family_delta"])
        self.assertFalse(gate["checks"]["nonnegative_families"])
        below = runner.decide_train_gate(
            "0.1", _family_values(negative="-0.000500000001")
        )
        self.assertFalse(below["checks"]["worst_family_delta"])

    def test_dev_comparison_is_strict(self):
        self.assertFalse(runner.decide_dev_gate("0.673182")["gate_passed"])
        self.assertTrue(runner.decide_dev_gate("0.673182000001")["gate_passed"])

    def test_nonfinite_missing_or_malformed_evidence_fails_closed(self):
        for value in ("nan", "inf", None, True):
            self.assertFalse(runner.decide_dev_gate(value)["gate_passed"])
        family_deltas = _family_values()
        family_deltas.pop(next(iter(family_deltas)))
        self.assertFalse(
            runner.decide_train_gate("0.1", family_deltas)["gate_passed"]
        )


def _safety_policy(p99=1.1, maximum=1.2, breaches=0, whole=True, holdout=True):
    return {
        "tiers": {
            tier: {
                "whole_split": {"budget_passed": whole},
                "bootstrap": {
                    "cap_breaches": breaches,
                    "cost_ratio": {"p99": p99, "max": maximum},
                },
                "family_holdouts": {
                    family: {"evaluated": True, "budget_passed": holdout}
                    for family in risk_validation.FAMILY_LABELS
                },
            }
            for tier in TIERS
        }
    }


class SafetyGateTest(unittest.TestCase):
    def test_matching_p99_and_max_with_zero_breaches_passes(self):
        report = _safety_policy()
        self.assertTrue(runner.decide_safety_gate(report, report)["gate_passed"])

    def test_any_worse_tail_or_budget_breach_fails(self):
        reference = _safety_policy()
        for candidate in (
            _safety_policy(p99=1.100000000001),
            _safety_policy(maximum=1.200000000001),
            _safety_policy(breaches=1),
            _safety_policy(whole=False),
            _safety_policy(holdout=False),
        ):
            self.assertFalse(
                runner.decide_safety_gate(reference, candidate)["gate_passed"]
            )

    def test_malformed_safety_evidence_fails_closed(self):
        self.assertFalse(runner.decide_safety_gate({}, {})["gate_passed"])


class AccessOrderingTest(unittest.TestCase):
    def test_train_failure_never_reaches_dev(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "evaluate_train", return_value={"gate_passed": False}
        ), mock.patch.object(
            runner, "_load_split_once", side_effect=AssertionError("Dev opened")
        ):
            report = runner.run_experiment(
                report_path=pathlib.Path(directory) / "report.json"
            )
        self.assertFalse(report["dev"]["accessed"])
        self.assertEqual(0, report["dev"]["evaluation_count"])
        self.assertEqual("safe-margin", report["decision"]["submission_default"])

    def test_dev_is_loaded_exactly_once_and_safety_reuses_it(self):
        sentinel = object()
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "evaluate_train", return_value={"gate_passed": True}
        ), mock.patch.object(
            runner, "_load_split_once", return_value=sentinel
        ) as loader, mock.patch.object(
            runner,
            "evaluate_dev_loaded",
            return_value={"accessed": True, "evaluation_count": 1, "gate_passed": True},
        ) as dev, mock.patch.object(
            runner,
            "evaluate_safety_loaded",
            return_value={"accessed": True, "evaluation_count": 1, "gate_passed": True},
        ) as safety:
            report = runner.run_experiment(
                report_path=pathlib.Path(directory) / "report.json"
            )
        loader.assert_called_once()
        dev.assert_called_once_with(sentinel)
        safety.assert_called_once_with(sentinel)
        self.assertTrue(report["decision"]["eligible"])
        self.assertEqual("safe-margin", report["decision"]["submission_default"])

    def test_protocol_and_json_output_are_frozen_and_byte_stable(self):
        self.assertEqual(
            runner.EXPECTED_PROTOCOL_SHA256,
            hashlib.sha256(runner.PROTOCOL_PATH.read_bytes()).hexdigest(),
        )
        runner.verify_protocol()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            value = {"z": 3, "a": {"eligible": False}}
            runner._atomic_json(path, value)
            first = path.read_bytes()
            runner._atomic_json(path, value)
            self.assertEqual(first, path.read_bytes())

    def test_existing_report_reemit_does_not_evaluate_any_split(self):
        value = {
            "report_type": runner.REPORT_TYPE,
            "protocol_sha256": runner.EXPECTED_PROTOCOL_SHA256,
            "experiment_id": runner.EXPERIMENT_ID,
            "decision": {"submission_default": "safe-margin"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"
            runner._atomic_json(path, value)
            first = path.read_bytes()
            with mock.patch.object(
                runner, "evaluate_train", side_effect=AssertionError("Train evaluated")
            ), mock.patch.object(
                runner, "_load_split_once", side_effect=AssertionError("Dev evaluated")
            ):
                runner.reemit_existing_report(path)
            self.assertEqual(first, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
