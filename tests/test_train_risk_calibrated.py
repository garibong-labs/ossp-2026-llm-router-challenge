# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the group-CV internals of the v2 trainer.

The trainer is development-only, but its honesty claims are load-bearing for
the recorded negative result: every internal validation split must be grouped
by source/task family, deterministic, and fail closed on dishonest requests.
"""

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


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = _load_module(
    "train_risk_calibrated", ROOT / "baselines/train_risk_calibrated.py"
)
np = trainer.np

GROUPS = (
    ["belebele-ko"] * 9
    + ["deepmind-mathematics"] * 8
    + ["ruletaker"] * 7
    + ["cruxeval"] * 6
    + ["gsm8k"] * 5
    + ["truthfulqa"] * 4
    + ["babilong"] * 3
    + ["hrmcr"] * 2
    + ["aime"] * 1
)


@unittest.skipIf(np is None, "requires NumPy")
class GroupFoldAssignmentTest(unittest.TestCase):
    def test_assignment_is_deterministic(self):
        first = trainer._group_fold_ids(GROUPS, 3)
        second = trainer._group_fold_ids(GROUPS, 3)
        self.assertTrue((first[0] == second[0]).all())
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_every_group_lands_in_exactly_one_validation_fold(self):
        fold_ids, mapping, fold_rows = trainer._group_fold_ids(GROUPS, 4)
        self.assertEqual(sorted(mapping), sorted(set(GROUPS)))
        for row, group in enumerate(GROUPS):
            self.assertEqual(fold_ids[row], mapping[group])
        self.assertEqual(sum(fold_rows), len(GROUPS))
        self.assertGreater(min(fold_rows), 0)

    def test_no_group_crosses_train_and_validation_in_any_fold(self):
        fold_ids, _mapping, _rows = trainer._group_fold_ids(GROUPS, 4)
        for fold in sorted(set(fold_ids.tolist())):
            validation_groups = {
                GROUPS[row]
                for row in range(len(GROUPS))
                if fold_ids[row] == fold
            }
            training_groups = {
                GROUPS[row]
                for row in range(len(GROUPS))
                if fold_ids[row] != fold
            }
            self.assertFalse(validation_groups & training_groups)

    def test_invalid_fold_requests_fail_closed(self):
        with self.assertRaises(ValueError):
            trainer._group_fold_ids(GROUPS, 1)
        with self.assertRaises(ValueError):
            trainer._group_fold_ids(GROUPS, len(set(GROUPS)) + 1)
        with self.assertRaises(ValueError):
            trainer._group_fold_ids([], 2)


@unittest.skipIf(np is None, "requires NumPy")
class GroupOofTest(unittest.TestCase):
    def setUp(self):
        rows = len(GROUPS)
        self.fold_ids, _mapping, _rows = trainer._group_fold_ids(GROUPS, 3)
        # Column 0 carries the row index so patched fits can report exactly
        # which rows they were trained on.
        self.matrix = np.column_stack(
            [np.arange(rows, dtype=np.float64), np.ones(rows)]
        )
        self.targets = np.linspace(-1.0, 1.0, rows)

    def test_every_row_gets_exactly_one_finite_oof_prediction(self):
        predictions = trainer._oof_predictions(
            self.matrix, self.targets, fold_ids=self.fold_ids, alpha=1.0
        )
        self.assertEqual(predictions.shape, self.targets.shape)
        self.assertTrue(np.isfinite(predictions).all())

    def test_ridge_oof_never_trains_on_the_validation_family(self):
        training_row_sets = []
        original_fit = trainer._fit_ridge
        original_predict = trainer._predict_ridge

        def fake_fit(matrix, targets, alpha):
            training_row_sets.append(frozenset(matrix[:, 0].astype(int).tolist()))
            return (None, None, None, None)

        def fake_predict(matrix, mean, scale, intercept, coefficients):
            trained = training_row_sets[-1]
            return np.asarray(
                [1.0 if int(row) in trained else 0.0 for row in matrix[:, 0]]
            )

        trainer._fit_ridge = fake_fit
        trainer._predict_ridge = fake_predict
        try:
            predictions = trainer._oof_predictions(
                self.matrix, self.targets, fold_ids=self.fold_ids, alpha=1.0
            )
        finally:
            trainer._fit_ridge = original_fit
            trainer._predict_ridge = original_predict
        # A 1.0 anywhere would mean a validation row sat in its own
        # training set; group folds make that impossible.
        self.assertEqual(float(predictions.max()), 0.0)
        for fold, trained in zip(sorted(set(self.fold_ids.tolist())),
                                 training_row_sets):
            validation_groups = {
                GROUPS[row]
                for row in range(len(GROUPS))
                if self.fold_ids[row] == fold
            }
            trained_groups = {GROUPS[row] for row in trained}
            self.assertFalse(validation_groups & trained_groups)

    def test_tree_oof_shares_the_same_group_fold_assignment(self):
        training_row_sets = []
        original_fit = trainer._fit_boosted_trees
        original_predict = trainer._predict_boosted_trees

        def fake_fit(X, y, **kwargs):
            training_row_sets.append(frozenset(X[:, 0].astype(int).tolist()))
            return ["model"]

        def fake_predict(X, model):
            trained = training_row_sets[-1]
            return np.asarray(
                [1.0 if int(row) in trained else 0.0 for row in X[:, 0]]
            )

        trainer._fit_boosted_trees = fake_fit
        trainer._predict_boosted_trees = fake_predict
        try:
            predictions = trainer._oof_boosted_trees(
                self.matrix, self.targets, self.fold_ids
            )
        finally:
            trainer._fit_boosted_trees = original_fit
            trainer._predict_boosted_trees = original_predict
        self.assertEqual(float(predictions.max()), 0.0)
        for fold, trained in zip(sorted(set(self.fold_ids.tolist())),
                                 training_row_sets):
            validation_rows = {
                row
                for row in range(len(GROUPS))
                if self.fold_ids[row] == fold
            }
            self.assertFalse(validation_rows & trained)

    def test_malformed_fold_assignments_fail_closed(self):
        with self.assertRaises(ValueError):
            trainer._oof_predictions(
                self.matrix,
                self.targets,
                fold_ids=np.zeros(len(GROUPS), dtype=np.int64),
                alpha=1.0,
            )
        with self.assertRaises(ValueError):
            trainer._oof_predictions(
                self.matrix,
                self.targets,
                fold_ids=self.fold_ids[:-1],
                alpha=1.0,
            )


@unittest.skipIf(np is None, "requires NumPy")
class GroupPartitionTest(unittest.TestCase):
    def test_two_way_split_is_disjoint_deterministic_and_complete(self):
        side_a, side_b = trainer._two_way_group_split(GROUPS)
        again = trainer._two_way_group_split(GROUPS)
        self.assertEqual((side_a, side_b), again)
        self.assertFalse(side_a & side_b)
        self.assertEqual(side_a | side_b, frozenset(GROUPS))
        self.assertTrue(side_a and side_b)

    def test_two_way_split_requires_at_least_two_groups(self):
        with self.assertRaises(ValueError):
            trainer._two_way_group_split(["only"] * 10)

    def test_calibration_comparison_uses_group_disjoint_sides(self):
        rows = len(GROUPS)
        rng_raw = np.linspace(-2.0, 2.0, rows)
        labels = (rng_raw + np.sin(np.arange(rows)) > 0).astype(np.float64)
        side_a, _side_b = trainer._two_way_group_split(GROUPS)
        mask = np.asarray([group in side_a for group in GROUPS])

        fit_sizes = {"platt": [], "isotonic": []}
        original_platt = trainer._fit_platt
        original_isotonic = trainer._fit_isotonic

        def spy_platt(raw, labels_):
            fit_sizes["platt"].append(len(raw))
            return original_platt(raw, labels_)

        def spy_isotonic(raw, labels_):
            fit_sizes["isotonic"].append(len(raw))
            return original_isotonic(raw, labels_)

        trainer._fit_platt = spy_platt
        trainer._fit_isotonic = spy_isotonic
        try:
            _chosen, diagnostics = trainer._choose_calibration(
                rng_raw, labels, mask
            )
        finally:
            trainer._fit_platt = original_platt
            trainer._fit_isotonic = original_isotonic
        self.assertIn(diagnostics["method"], ("platt", "isotonic"))
        side_sizes = {int(mask.sum()), int((~mask).sum())}
        # Comparison fits saw exactly one side each; the final fit (of the
        # chosen method only) saw every row, strictly after the comparison.
        self.assertEqual(set(fit_sizes["platt"][:2]), side_sizes)
        self.assertEqual(set(fit_sizes["isotonic"][:2]), side_sizes)
        final = fit_sizes[diagnostics["method"]]
        self.assertEqual(final[-1], rows)
        other = "isotonic" if diagnostics["method"] == "platt" else "platt"
        self.assertEqual(len(fit_sizes[other]), 2)

    def test_calibration_comparison_rejects_degenerate_masks(self):
        raw = np.linspace(0.0, 1.0, len(GROUPS))
        labels = np.zeros(len(GROUPS))
        with self.assertRaises(ValueError):
            trainer._choose_calibration(
                raw, labels, np.ones(len(GROUPS), dtype=bool)
            )
        with self.assertRaises(ValueError):
            trainer._choose_calibration(raw, labels, np.zeros(3, dtype=bool))

    def test_conformal_factor_ignores_the_evaluation_side(self):
        mask = np.asarray([True] * 20 + [False] * 20)
        residuals = np.concatenate(
            [np.full(20, np.log(1.1)), np.full(20, np.log(2.0))]
        )
        factor, coverage = trainer._conformal_upper(residuals, mask, 0.85)
        self.assertAlmostEqual(factor, 1.1, places=12)
        self.assertEqual(coverage, 0.0)
        # Changing only the evaluation side must not move the factor.
        shifted = residuals.copy()
        shifted[~mask] = np.log(1.05)
        factor_again, coverage_again = trainer._conformal_upper(
            shifted, mask, 0.85
        )
        self.assertAlmostEqual(factor_again, 1.1, places=12)
        self.assertEqual(coverage_again, 1.0)

    def test_conformal_factor_is_floored_at_one(self):
        mask = np.asarray([True] * 10 + [False] * 10)
        residuals = np.full(20, np.log(0.5))
        factor, coverage = trainer._conformal_upper(residuals, mask, 0.85)
        self.assertEqual(factor, 1.0)
        self.assertEqual(coverage, 1.0)

    def test_conformal_masks_fail_closed(self):
        residuals = np.zeros(10)
        with self.assertRaises(ValueError):
            trainer._conformal_upper(residuals, np.ones(10, dtype=bool), 0.85)
        with self.assertRaises(ValueError):
            trainer._conformal_upper(residuals, np.ones(4, dtype=bool), 0.85)

    def test_symmetric_conformal_ships_the_more_conservative_direction(self):
        mask = np.asarray([True] * 20 + [False] * 20)
        residuals = np.concatenate(
            [np.full(20, np.log(1.1)), np.full(20, np.log(2.0))]
        )
        factor, measured, diagnostics = trainer._conformal_upper_symmetric(
            residuals, mask, 0.85
        )
        # Side B's quantile (2.0) beats side A's (1.1); coverage must then
        # come from side A, which the shipped factor fully covers.
        self.assertAlmostEqual(factor, 2.0, places=12)
        self.assertEqual(measured, 1.0)
        self.assertEqual(diagnostics["source_side"], "side_b")
        self.assertAlmostEqual(diagnostics["side_a_factor"], 1.1, places=12)
        self.assertEqual(
            diagnostics["side_a_heldout_coverage_on_side_b"], 0.0
        )

    def test_symmetric_conformal_tie_reports_the_worse_coverage(self):
        mask = np.asarray([True] * 20 + [False] * 20)
        residuals = np.full(40, np.log(1.5))
        # Two side-B rows above the shared quantile: both directions still
        # produce the same 1.5 factor (rank 17 of 20), but side B's held-out
        # coverage drops to 0.9 while side A's stays 1.0.
        residuals[-2:] = np.log(3.0)
        factor, measured, diagnostics = trainer._conformal_upper_symmetric(
            residuals, mask, 0.85
        )
        self.assertEqual(diagnostics["source_side"], "tie")
        self.assertAlmostEqual(factor, 1.5, places=12)
        self.assertEqual(measured, 0.9)


class FrozenTrainingReportTest(unittest.TestCase):
    """The shipped report must prove its splits were group-disjoint."""

    REPORT = ROOT / "baselines/risk-calibrated-train-report.v2.json"

    def setUp(self):
        self.report = json.loads(self.REPORT.read_text(encoding="utf-8"))

    def test_group_cv_partitions_every_row_into_disjoint_folds(self):
        group_cv = self.report["group_cv"]
        counts = group_cv["group_counts"]
        mapping = group_cv["group_to_fold"]
        self.assertEqual(sorted(counts), sorted(mapping))
        self.assertEqual(
            sum(counts.values()),
            self.report["training_summary"]["num_episodes"],
        )
        fold_rows = [0] * self.report["training_summary"]["folds"]
        for group, fold in mapping.items():
            fold_rows[fold] += counts[group]
        self.assertEqual(fold_rows, group_cv["fold_row_counts"])
        self.assertGreater(min(fold_rows), 0)

    def test_calibration_and_conformal_partitions_are_group_disjoint(self):
        counts = self.report["group_cv"]["group_counts"]
        for section, left, right in (
            ("calibration_partition", "side_a_groups", "side_b_groups"),
            ("conformal_partition", "side_a_groups", "side_b_groups"),
        ):
            record = self.report[section]
            side_a = set(record[left])
            side_b = set(record[right])
            self.assertFalse(side_a & side_b)
            self.assertEqual(side_a | side_b, set(counts))
            self.assertTrue(side_a and side_b)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
