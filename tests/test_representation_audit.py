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

    def test_candidate_prompt_tail_beyond_bound_cannot_change_features(self):
        bound = representation_features.MAX_FIELD_CHARACTERS
        seed = "Question: bounded?\n"
        prefix = seed + "x" * (bound - len(seed))
        left = Episode("left", prompt=prefix + " tail alpha 123 !!!")
        right = Episode("right", prompt=prefix + " tail beta ``` ???")
        for name in representation_features.REPRESENTATIONS[1:]:
            self.assertEqual(
                representation_features.representation_vector(left, name),
                representation_features.representation_vector(right, name),
            )
        self.assertNotEqual(
            representation_features.representation_vector(left, "A-current-dense-wordhash"),
            representation_features.representation_vector(right, "A-current-dense-wordhash"),
        )

    def test_candidate_message_tails_are_bounded_per_field(self):
        bound = representation_features.MAX_FIELD_CHARACTERS
        prefix = "m" * bound
        left = Episode(
            "left",
            messages=(Message("system", prefix + "alpha"), Message("user", prefix + "123")),
        )
        right = Episode(
            "right",
            messages=(Message("system", prefix + "beta"), Message("user", prefix + "???")),
        )
        for name in representation_features.REPRESENTATIONS[1:]:
            self.assertEqual(
                representation_features.representation_vector(left, name),
                representation_features.representation_vector(right, name),
            )


@unittest.skipIf(audit is None, "requires the pinned training-only NumPy")
class AuditHonestyTest(unittest.TestCase):
    @staticmethod
    def _gate_result(runtime, value=0.01):
        all_labels = ("fast", "balanced", "premium")
        per_family = {
            family: {
                "oof_correlation": 0.1,
                "selected_set": {
                    label: {"realized_incremental_gain": value}
                    for label in all_labels
                },
            }
            for family in audit.risk_validation.FAMILY_LABELS
        }
        metrics = {
            step: {
                "oof_correlation": 0.1,
                "positive_family_correlations": 9,
                "selected_set": {
                    label: {"realized_incremental_gain": value}
                    for label in labels
                },
                "per_held_out_family": per_family,
            }
            for step, labels in audit.SPEND_LABELS.items()
        }
        return {"runtime_feasibility": runtime, "metrics": metrics}

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

    def test_candidate_gate_rejects_false_or_missing_bound_metadata(self):
        valid_runtime = {
            "passed": True,
            "candidate_input_bound_required": True,
            "candidate_input_bound_enforced": True,
            "candidate_input_bound_probe": True,
            "characters_per_field_bound": representation_features.MAX_FIELD_CHARACTERS,
        }
        reference = self._gate_result(valid_runtime, value=0.0)
        self.assertTrue(
            audit.representation_gate(
                "B-expanded-structural",
                self._gate_result(valid_runtime),
                reference,
            )["passed"]
        )
        for mutation in (
            {"candidate_input_bound_enforced": False},
            {"candidate_input_bound_probe": False},
            {"candidate_input_bound_required": False},
            {"characters_per_field_bound": None},
            {"characters_per_field_bound": representation_features.MAX_FIELD_CHARACTERS + 1},
        ):
            runtime = dict(valid_runtime)
            runtime.update(mutation)
            gate = audit.representation_gate(
                "B-expanded-structural", self._gate_result(runtime), reference
            )
            self.assertFalse(gate["passed"])
            self.assertEqual("runtime_feasibility", gate["failed_checks"][0]["check"])


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

    def test_report_runtime_bounds_are_representation_specific(self):
        report = json.loads(self.REPORT.read_text(encoding="utf-8"))
        records = report["train_representations"]
        reference = records["A-current-dense-wordhash"]["runtime_feasibility"]
        self.assertIsNone(reference["characters_per_field_bound"])
        self.assertFalse(reference["candidate_input_bound_required"])
        self.assertFalse(reference["candidate_input_bound_enforced"])
        self.assertFalse(reference["candidate_input_bound_probe"])
        for name in representation_features.REPRESENTATIONS[1:]:
            runtime = records[name]["runtime_feasibility"]
            self.assertEqual(
                representation_features.MAX_FIELD_CHARACTERS,
                runtime["characters_per_field_bound"],
            )
            self.assertTrue(runtime["candidate_input_bound_required"])
            self.assertTrue(runtime["candidate_input_bound_enforced"])
            self.assertTrue(runtime["candidate_input_bound_probe"])
        self.assertEqual(
            ["source/task-family label", "episode_id", "outcome", "Dev outcome", "split identity", "row position"],
            report["protocol"]["forbidden_runtime_features"],
        )


if __name__ == "__main__":
    unittest.main()
