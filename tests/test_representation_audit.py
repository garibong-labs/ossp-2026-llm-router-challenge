# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused honesty and fail-closed tests for the representation audit."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools", ROOT / "baselines"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from ossp_router.protocol import Episode, Message  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


representation_features = _load(
    "representation_features", ROOT / "baselines/representation_features.py"
)
try:
    import numpy  # noqa: F401
except ImportError:
    audit = None
else:
    audit = _load("representation_audit", ROOT / "tools/representation_audit.py")


class RepresentationFeatureTest(unittest.TestCase):
    def test_all_candidate_shapes_are_fixed_and_deterministic(self):
        episode = Episode(
            "ignored-id",
            messages=(
                Message("system", "Use the supplied context only."),
                Message("user", "Context: alpha beta.\nQuestion: What follows?"),
            ),
        )
        expected = {
            "A-current-dense-wordhash": 270,
            "B-expanded-structural": 36,
            "C-semantic-proxy": 270,
            "D-structural-semantic": 292,
        }
        for name, size in expected.items():
            first = representation_features.representation_vector(episode, name)
            second = representation_features.representation_vector(episode, name)
            self.assertEqual(size, len(first))
            self.assertEqual(first, second)

    def test_episode_id_is_not_a_feature(self):
        left = Episode("train-source-secret", prompt="Question: 2 + 2?")
        right = Episode("dev-other-family", prompt="Question: 2 + 2?")
        for name in representation_features.REPRESENTATIONS:
            self.assertEqual(
                representation_features.representation_vector(left, name),
                representation_features.representation_vector(right, name),
            )

    def test_structural_features_derive_roles_and_context_boundary(self):
        episode = Episode(
            "e",
            messages=(
                Message("system", "policy"),
                Message("user", "Long context here.\nQuestion: answer this?"),
                Message("assistant", "prior answer"),
            ),
        )
        vector = representation_features.expanded_structural_vector(episode)
        values = dict(zip(representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES, vector))
        self.assertEqual(1.0, values["has_explicit_system_field"])
        self.assertEqual(1.0, values["has_role_transition"])
        self.assertEqual(1.0, values["has_context_question_boundary"])
        self.assertGreater(values["context_fraction"], 0.0)


@unittest.skipIf(audit is None, "requires the pinned training-only NumPy")
class AuditHonestyTest(unittest.TestCase):
    def test_leave_one_family_out_is_disjoint_and_complete(self):
        families = tuple(
            family
            for family in audit.risk_validation.FAMILY_LABELS
            for _ in range(2)
        )
        fold_ids, mapping = audit._lofo_ids(families)
        self.assertEqual(9, len(set(fold_ids.tolist())))
        for family, fold in mapping.items():
            validation = {families[index] for index in range(len(families)) if fold_ids[index] == fold}
            training = {families[index] for index in range(len(families)) if fold_ids[index] != fold}
            self.assertEqual({family}, validation)
            self.assertFalse(validation & training)

    def test_leave_one_family_out_fails_on_missing_group(self):
        with self.assertRaises(ValueError):
            audit._lofo_ids(audit.risk_validation.FAMILY_LABELS[:-1])

    def test_winner_selection_has_no_dev_argument_or_state(self):
        def record(value):
            return {
                "metrics": {
                    step: {
                        "selected_set": {
                            label: {"realized_incremental_gain": value}
                            for label in audit.SPEND_LABELS[step]
                        }
                    }
                    for step in audit.STEP_NAMES
                }
            }

        self.assertEqual(
            "candidate",
            audit.select_frozen_winner({"reference": record(0.0), "candidate": record(0.1)}),
        )

    def test_gate_fails_closed_when_signal_is_unstable(self):
        all_labels = ("fast", "balanced", "premium")
        per_family = {
            family: {
                "oof_correlation": -0.1,
                "selected_set": {
                    label: {"realized_incremental_gain": 0.0}
                    for label in all_labels
                },
            }
            for family in audit.risk_validation.FAMILY_LABELS
        }
        metrics = {
            step: {
                "oof_correlation": -0.01,
                "positive_family_correlations": 0,
                "selected_set": {
                    label: {"realized_incremental_gain": 0.0}
                    for label in labels
                },
                "per_held_out_family": per_family,
            }
            for step, labels in audit.SPEND_LABELS.items()
        }
        result = {"runtime_feasibility": {"passed": True}, "metrics": metrics}
        gate = audit.representation_gate("candidate", result, result)
        self.assertFalse(gate["passed"])
        self.assertTrue(gate["failed_checks"])


class FrozenReportTest(unittest.TestCase):
    REPORT = ROOT / "baselines/representation-audit-report.v1.json"

    def test_report_preserves_negative_result_and_safe_default(self):
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        self.assertEqual("ossp-representation-family-audit-v1", report["report_type"])
        self.assertFalse(report["decision"]["candidate_adopted"])
        self.assertEqual("safe-margin", report["decision"]["submission_default"])
        self.assertTrue(
            report["frozen_dev_evaluation"]["selection_frozen_before_dev_load"]
        )
        self.assertFalse(report["frozen_dev_evaluation"]["dev_used_for_selection"])

    def test_every_train_family_has_held_out_metrics(self):
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        expected = {
            "aime", "babilong", "belebele-ko", "cruxeval",
            "deepmind-mathematics", "gsm8k", "hrmcr", "ruletaker",
            "truthfulqa",
        }
        for representation in report["train_representations"].values():
            for step in ("ax31-light->ax31", "ax31->axk1-think"):
                self.assertEqual(
                    expected,
                    set(representation["metrics"][step]["per_held_out_family"]),
                )


if __name__ == "__main__":
    unittest.main()
