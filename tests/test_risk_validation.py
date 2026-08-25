# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the development-only risk-validation harness."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


risk_validation = _load_module(
    "risk_validation", ROOT / "tools/risk_validation.py"
)

MATERIALIZED = ROOT / "data/materialized/dev/inputs.json"


class PromptFamilyClassificationTest(unittest.TestCase):
    def test_each_family_rule_matches_a_representative_prompt(self):
        cases = {
            "cruxeval": 'def f(text):\n    return text\n\nassert f(??) == ""',
            "babilong": "word " * 700 + "Where is the apple?",
            "truthfulqa": "Question: What happens if you break a mirror?\nA. X",
            "belebele-ko": (
                "위성 전화는 실외에서 사용해야 합니다. 지역 통신사가 안내합니다."
                "\nQuestion: 무엇입니까?\nA. 첫째\nB. 둘째"
            ),
            "hrmcr": "갑은 을보다 두 살 많고 병은 갑보다 어리다. 나이 순서를 구하시오.",
            "ruletaker": (
                "The cat likes the dog. The dog is round. The cat is blue. "
                "If someone likes the dog then they are cold. "
                "The dog does not see the cat."
            ),
            "gsm8k": (
                "Mark buys 12 cars for $20,000 each. He pays 10% tax. "
                "How much does he spend in total?"
            ),
        }
        for family, prompt in cases.items():
            self.assertEqual(
                family,
                risk_validation.classify_prompt(prompt),
                f"expected {family}",
            )

    @unittest.skipUnless(
        MATERIALIZED.is_file(), "requires materialized public data"
    )
    def test_reconstructed_families_partition_the_public_dev_split(self):
        from ossp_router.protocol import load_input

        inputs = load_input(MATERIALIZED)
        families = risk_validation.reconstruct_families("dev", inputs)
        self.assertEqual(len(inputs.episodes), len(families))
        self.assertLessEqual(
            set(families), set(risk_validation.FAMILY_LABELS)
        )
        counts = {
            family: families.count(family)
            for family in risk_validation.FAMILY_LABELS
        }
        # The selection files pin these two exactly.
        self.assertEqual(12, counts["aime"])
        self.assertEqual(153, counts["deepmind-mathematics"])
        self.assertTrue(all(count > 0 for count in counts.values()))


def _tier_report(*, p99, maximum, breaches=0, whole=1.05, holdout_ratio=1.05):
    return {
        "whole_split": {
            "cost_ratio": whole,
            "budget_passed": True,
            "quality_score": 0.65,
            "model_counts": {},
            "single_episode_concentration": 0.05,
        },
        "bootstrap": {
            "cost_ratio": {"p99": p99, "max": maximum},
            "cap_breaches": breaches,
            "resamples": 5000,
        },
        "family_holdouts": {
            family: {
                "evaluated": True,
                "budget_passed": True,
                "cost_ratio": holdout_ratio,
            }
            for family in risk_validation.FAMILY_LABELS
        },
    }


def _policy_report(**kwargs):
    return {"tiers": {tier: _tier_report(**kwargs) for tier in ("fast", "balanced", "premium")}}


class GateLogicTest(unittest.TestCase):
    def test_a_candidate_matching_the_reference_passes(self):
        reference = _policy_report(p99=1.10, maximum=1.15)
        candidate = _policy_report(p99=1.10, maximum=1.15)
        gate = risk_validation.gate_candidate(reference, candidate)
        self.assertTrue(gate["passed"])
        self.assertEqual([], gate["failed_checks"])

    def test_a_single_resample_breach_fails_the_gate(self):
        reference = _policy_report(p99=1.10, maximum=1.15)
        candidate = _policy_report(p99=1.10, maximum=1.15, breaches=1)
        gate = risk_validation.gate_candidate(reference, candidate)
        self.assertFalse(gate["passed"])

    def test_p99_beyond_the_documented_tolerance_fails(self):
        reference = _policy_report(p99=1.10, maximum=1.15)
        candidate = _policy_report(p99=1.107, maximum=1.15)
        gate = risk_validation.gate_candidate(
            reference, candidate, tolerance=0.005
        )
        self.assertFalse(gate["passed"])
        candidate = _policy_report(p99=1.104, maximum=1.15)
        gate = risk_validation.gate_candidate(
            reference, candidate, tolerance=0.005
        )
        self.assertTrue(gate["passed"])

    def test_an_unevaluated_family_holdout_is_indeterminate_and_fails(self):
        reference = _policy_report(p99=1.10, maximum=1.15)
        candidate = _policy_report(p99=1.10, maximum=1.15)
        candidate["tiers"]["fast"]["family_holdouts"]["gsm8k"] = {
            "evaluated": False,
            "held_out": 0,
        }
        gate = risk_validation.gate_candidate(reference, candidate)
        self.assertFalse(gate["passed"])
        failed = [item["check"] for item in gate["failed_checks"]]
        self.assertIn("fast.holdout.gsm8k", failed)

    def test_a_worse_family_holdout_ratio_fails(self):
        reference = _policy_report(p99=1.10, maximum=1.15, holdout_ratio=1.05)
        candidate = _policy_report(p99=1.10, maximum=1.15, holdout_ratio=1.08)
        gate = risk_validation.gate_candidate(reference, candidate)
        self.assertFalse(gate["passed"])


class DeterminismTest(unittest.TestCase):
    def test_bootstrap_indices_are_seed_deterministic(self):
        from stress_safe_margin import bootstrap_indices

        first = bootstrap_indices(100, 5, risk_validation.DEFAULT_SEED)
        second = bootstrap_indices(100, 5, risk_validation.DEFAULT_SEED)
        self.assertEqual(first, second)

    def test_frozen_gate_configuration_values(self):
        self.assertEqual(5000, risk_validation.DEFAULT_RESAMPLES)
        self.assertEqual(20260825, risk_validation.DEFAULT_SEED)
        self.assertEqual(0.005, risk_validation.HEADROOM_TOLERANCE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
