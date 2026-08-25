# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Audit prompt representations with source/task-family-disjoint evidence.

Representation and ridge selection use public Train only.  The selected
diagnostic leader is frozen before this tool loads public Dev outcomes, which
are evaluated once and never participate in selection.  A representation is
adoptable only if the predeclared Train gate passes; integration still requires
the unchanged 5,000-resample safety gate and weighted Dev champion threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools", ROOT / "baselines"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402

import representation_features as features  # noqa: E402
import risk_validation  # noqa: E402
import train_risk_calibrated as trainer  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    InputBatch,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
)


REPORT_TYPE = "ossp-representation-family-audit-v1"
ALPHAS = (100.0, 1000.0, 3000.0)
TRAIN_CORRELATION_FLOOR = 0.02
MIN_POSITIVE_FAMILIES = 6
SELECTED_GAIN_MARGIN = 0.002
FAMILY_GAIN_FLOOR = -0.005
MAX_FEATURES = 1024
MAX_ESTIMATED_ARTIFACT_BYTES = 1_000_000
DEV_CHAMPION_THRESHOLD = 0.690000
SPEND_GOALS = dict(trainer.SPEND_GOALS)

STEP_NAMES = ("ax31-light->ax31", "ax31->axk1-think")
SPEND_LABELS = {
    STEP_NAMES[0]: ("fast", "balanced"),
    STEP_NAMES[1]: ("premium",),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _correlation(predicted: Any, realized: Any) -> float:
    if len(predicted) < 2 or np.std(predicted) <= 1e-15 or np.std(realized) <= 1e-15:
        return 0.0
    value = float(np.corrcoef(predicted, realized)[0, 1])
    return value if math.isfinite(value) else 0.0


def _matrix(inputs: InputBatch, representation: str) -> Tuple[Any, Mapping[str, Any]]:
    rows = [features.representation_vector(episode, representation) for episode in inputs.episodes]
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{representation} produced a malformed feature matrix")
    probes = [features.representation_vector(episode, representation) for episode in inputs.episodes[:8]]
    deterministic = all(tuple(matrix[index]) == tuple(probe) for index, probe in enumerate(probes))
    artifact_bytes = int(matrix.shape[1] * 5 * 12 + 16_384)
    feasible = (
        deterministic
        and matrix.shape[1] <= MAX_FEATURES
        and artifact_bytes <= MAX_ESTIMATED_ARTIFACT_BYTES
    )
    return matrix, {
        "deterministic_probe": deterministic,
        "feature_count": int(matrix.shape[1]),
        "full_split_extraction_completed": True,
        "characters_per_field_bound": features.MAX_FIELD_CHARACTERS,
        "estimated_five_head_artifact_bytes": artifact_bytes,
        "limits": {
            "max_features": MAX_FEATURES,
            "official_runtime_seconds_if_integrated": 90,
            "max_estimated_artifact_bytes": MAX_ESTIMATED_ARTIFACT_BYTES,
        },
        "passed": feasible,
        "runtime_dependencies": ["python-stdlib"],
        "network_or_external_weights": False,
    }


def _lofo_ids(families: Sequence[str]) -> Tuple[Any, Mapping[str, int]]:
    names = sorted(set(families))
    if tuple(names) != tuple(sorted(risk_validation.FAMILY_LABELS)):
        raise ValueError("Train must contain exactly the nine reconstructed families")
    mapping = {family: index for index, family in enumerate(names)}
    return np.asarray([mapping[family] for family in families], dtype=np.int64), mapping


def _budget(
    step_index: int,
    spend_label: str,
    costs: Any,
    rows: Optional[Any] = None,
) -> Tuple[float, float]:
    if rows is None:
        rows = np.ones(len(costs), dtype=bool)
    light_total = float(costs[rows, 0].sum())
    if step_index == 0:
        return (SPEND_GOALS[spend_label] - 1.0) * light_total, light_total
    ax31_total = float(costs[rows, 1].sum())
    return max(0.0, SPEND_GOALS[spend_label] * light_total - ax31_total), light_total


def _selected_metrics(
    predicted_gain: Any,
    predicted_increment: Any,
    realized_gain: Any,
    realized_increment: Any,
    budget: float,
    light_total: float,
) -> Mapping[str, Any]:
    order = np.argsort(-(predicted_gain / np.maximum(predicted_increment, 1e-12)))
    chosen = []
    budget_used = 0.0
    realized_cost = 0.0
    for raw_index in order:
        index = int(raw_index)
        increment = float(realized_increment[index])
        conservative_increment = max(0.0, increment)
        if (
            predicted_gain[index] <= 0.0
            or budget_used + conservative_increment > budget
        ):
            continue
        chosen.append(index)
        budget_used += conservative_increment
        realized_cost += increment
    gain = math.fsum(float(realized_gain[index]) for index in chosen)
    return {
        "selected_count": len(chosen),
        "realized_incremental_gain": gain / len(predicted_gain),
        "realized_incremental_cost": realized_cost,
        "incremental_cost_over_light": realized_cost / max(light_total, 1e-12),
        "conservative_budget_used": budget_used,
        "frozen_incremental_budget": budget,
    }


def _all_metrics(
    gain_predictions: Any,
    increment_predictions: Any,
    gains: Any,
    increments: Any,
    costs: Any,
    families: Sequence[str],
) -> Mapping[str, Any]:
    result: Dict[str, Any] = {}
    for step_index, step in enumerate(STEP_NAMES):
        overall_selected = {}
        for label in SPEND_LABELS[step]:
            budget, light_total = _budget(step_index, label, costs)
            overall_selected[label] = _selected_metrics(
                gain_predictions[:, step_index], increment_predictions[:, step_index],
                gains[:, step_index], increments[:, step_index], budget, light_total,
            )
        per_family = {}
        for family in sorted(set(families)):
            mask = np.asarray([value == family for value in families], dtype=bool)
            selected = {}
            for label in SPEND_LABELS[step]:
                budget, light_total = _budget(step_index, label, costs, mask)
                selected[label] = _selected_metrics(
                    gain_predictions[mask, step_index], increment_predictions[mask, step_index],
                    gains[mask, step_index], increments[mask, step_index], budget, light_total,
                )
            per_family[family] = {
                "rows": int(mask.sum()),
                "oof_correlation": _correlation(gain_predictions[mask, step_index], gains[mask, step_index]),
                "selected_set": selected,
            }
        result[step] = {
            "oof_correlation": _correlation(gain_predictions[:, step_index], gains[:, step_index]),
            "realized_gain_mean": float(gains[:, step_index].mean()),
            "realized_incremental_cost_mean": float(increments[:, step_index].mean()),
            "selected_set": overall_selected,
            "positive_family_correlations": sum(item["oof_correlation"] > 0.0 for item in per_family.values()),
            "per_held_out_family": per_family,
        }
    return result


def _select_gain_alpha(
    matrix: Any,
    gains: Any,
    predicted_increments: Any,
    increments: Any,
    costs: Any,
    fold_ids: Any,
) -> Tuple[float, Any, Mapping[str, float]]:
    best = None
    diagnostics = {}
    for alpha in ALPHAS:
        predictions = trainer._oof_predictions(matrix, gains, fold_ids=fold_ids, alpha=alpha)
        objective = 0.0
        for step_index, step in enumerate(STEP_NAMES):
            for label in SPEND_LABELS[step]:
                budget, light_total = _budget(step_index, label, costs)
                metrics = _selected_metrics(
                    predictions[:, step_index],
                    predicted_increments[:, step_index],
                    gains[:, step_index], increments[:, step_index], budget, light_total,
                )
                objective += metrics["realized_incremental_gain"]
        diagnostics[format(alpha, ".12g")] = objective
        rank = (objective, -alpha)
        if best is None or rank > best[0]:
            best = (rank, alpha, predictions)
    assert best is not None
    return best[1], best[2], diagnostics


def _predicted_increments(log_cost_predictions: Any) -> Any:
    predicted_costs = np.exp(log_cost_predictions)
    return np.column_stack(
        (
            np.maximum(
                predicted_costs[:, 1] - predicted_costs[:, 0],
                predicted_costs[:, 0] * 0.5,
            ),
            np.maximum(
                predicted_costs[:, 2] - predicted_costs[:, 1],
                predicted_costs[:, 1] * 0.5,
            ),
        )
    )


def _select_cost_alpha(
    matrix: Any, costs: Any, fold_ids: Any
) -> Tuple[float, Any, Mapping[str, float]]:
    targets = np.log(costs)
    best = None
    diagnostics = {}
    for alpha in ALPHAS:
        raw = trainer._oof_predictions(matrix, targets, fold_ids=fold_ids, alpha=alpha)
        mse = float(np.mean((raw - targets) ** 2))
        diagnostics[format(alpha, ".12g")] = mse
        rank = (mse, alpha)
        if best is None or rank < best[0]:
            best = (rank, alpha, _predicted_increments(raw))
    assert best is not None
    return best[1], best[2], diagnostics


def evaluate_train_representation(
    matrix: Any,
    runtime: Mapping[str, Any],
    gains: Any,
    increments: Any,
    costs: Any,
    families: Sequence[str],
    fold_ids: Any,
) -> Mapping[str, Any]:
    cost_alpha, increment_oof, cost_trace = _select_cost_alpha(
        matrix, costs, fold_ids
    )
    gain_alpha, gain_oof, gain_trace = _select_gain_alpha(
        matrix, gains, increment_oof, increments, costs, fold_ids
    )
    return {
        "feature_count": int(matrix.shape[1]),
        "selected_alpha_gain": gain_alpha,
        "selected_alpha_incremental_cost": cost_alpha,
        "alpha_objectives": {"selected_set_gain_sum": gain_trace, "log_incremental_cost_mse": cost_trace},
        "runtime_feasibility": runtime,
        "metrics": _all_metrics(gain_oof, increment_oof, gains, increments, costs, families),
    }


def representation_gate(name: str, result: Mapping[str, Any], reference: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = []

    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})

    add("runtime_feasibility", result["runtime_feasibility"]["passed"], "bounded stdlib extraction and artifact estimate")
    for step in STEP_NAMES:
        metrics = result["metrics"][step]
        add(f"{step}.correlation", metrics["oof_correlation"] >= TRAIN_CORRELATION_FLOOR, f"{metrics['oof_correlation']:.6f} >= {TRAIN_CORRELATION_FLOOR:.6f}")
        add(f"{step}.positive_families", metrics["positive_family_correlations"] >= MIN_POSITIVE_FAMILIES, f"{metrics['positive_family_correlations']} >= {MIN_POSITIVE_FAMILIES}")
        family_floor = min(
            item["selected_set"][label]["realized_incremental_gain"]
            for item in metrics["per_held_out_family"].values()
            for label in SPEND_LABELS[step]
        )
        add(f"{step}.family_gain_floor", family_floor >= FAMILY_GAIN_FLOOR, f"{family_floor:.6f} >= {FAMILY_GAIN_FLOOR:.6f}")
        for label in SPEND_LABELS[step]:
            value = metrics["selected_set"][label]["realized_incremental_gain"]
            baseline = reference["metrics"][step]["selected_set"][label]["realized_incremental_gain"]
            add(f"{step}.{label}.beats_current", value >= baseline + SELECTED_GAIN_MARGIN, f"{value:.6f} >= {baseline + SELECTED_GAIN_MARGIN:.6f}")
    if name == features.REPRESENTATIONS[0]:
        add("new_representation", False, "the current representation is the audit reference, not a new candidate")
    return {"passed": all(item["passed"] for item in checks), "checks": checks, "failed_checks": [item for item in checks if not item["passed"]]}


def select_frozen_winner(
    results: Mapping[str, Mapping[str, Any]],
    names: Optional[Sequence[str]] = None,
) -> str:
    """Choose the diagnostic leader from Train records only."""

    def score(name: str) -> Tuple[float, str]:
        total = sum(
            results[name]["metrics"][step]["selected_set"][label]["realized_incremental_gain"]
            for step in STEP_NAMES for label in SPEND_LABELS[step]
        )
        return total, name

    candidates = tuple(names) if names is not None else tuple(results)
    if not candidates:
        raise ValueError("cannot select a representation from an empty set")
    return max(candidates, key=score)


def _fit_predict(matrix_train: Any, matrix_dev: Any, targets: Any, alpha: float) -> Any:
    mean, scale, intercept, coefficients = trainer._fit_ridge(matrix_train, targets, alpha)
    return trainer._predict_ridge(matrix_dev, mean, scale, intercept, coefficients)


def evaluate_frozen_dev(
    name: str,
    train_matrix: Any,
    dev_inputs: InputBatch,
    train_gains: Any,
    train_costs: Any,
    dev_gains: Any,
    dev_increments: Any,
    dev_costs: Any,
    dev_families: Sequence[str],
    train_result: Mapping[str, Any],
) -> Mapping[str, Any]:
    dev_matrix, runtime = _matrix(dev_inputs, name)
    gain_predictions = _fit_predict(train_matrix, dev_matrix, train_gains, train_result["selected_alpha_gain"])
    log_cost_predictions = _fit_predict(
        train_matrix, dev_matrix, np.log(train_costs),
        train_result["selected_alpha_incremental_cost"],
    )
    return {
        "representation": name,
        "selection_frozen_before_dev_load": True,
        "dev_used_for_selection": False,
        "runtime_feasibility": runtime,
        "metrics": _all_metrics(
            gain_predictions, _predicted_increments(log_cost_predictions), dev_gains,
            dev_increments, dev_costs, dev_families,
        ),
    }


def _tables(input_path: Path, outcomes_path: Path, policy: RoutingPolicy):
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    _unused, scores, costs = trainer._training_tables(inputs, outcomes, policy, 16)
    gains = np.column_stack((scores[:, 1] - scores[:, 0], scores[:, 2] - scores[:, 1]))
    increments = np.column_stack((costs[:, 1] - costs[:, 0], costs[:, 2] - costs[:, 1]))
    return inputs, gains, increments, costs


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_audit(
    train_input: Path,
    train_outcomes: Path,
    dev_input: Path,
    dev_outcomes: Path,
    report_path: Path,
) -> Mapping[str, Any]:
    policy = load_bundled_policy()
    train_inputs, train_gains, train_increments, train_costs = _tables(train_input, train_outcomes, policy)
    train_families = risk_validation.reconstruct_families("train", train_inputs)
    fold_ids, family_to_fold = _lofo_ids(train_families)
    matrices = {}
    train_results = {}
    for name in features.REPRESENTATIONS:
        matrix, runtime = _matrix(train_inputs, name)
        matrices[name] = matrix
        train_results[name] = evaluate_train_representation(
            matrix, runtime, train_gains, train_increments, train_costs,
            train_families, fold_ids,
        )
    reference = train_results[features.REPRESENTATIONS[0]]
    for name, result in train_results.items():
        result["adoption_gate"] = representation_gate(name, result, reference)
    eligible = [
        name
        for name in features.REPRESENTATIONS[1:]
        if train_results[name]["adoption_gate"]["passed"]
    ]
    frozen_winner = select_frozen_winner(
        train_results, eligible if eligible else None
    )
    train_gate_passed = train_results[frozen_winner]["adoption_gate"]["passed"]

    # Dev outcomes are intentionally loaded only after all Train-only choices
    # above have been frozen.
    dev_inputs, dev_gains, dev_increments, dev_costs = _tables(dev_input, dev_outcomes, policy)
    dev_families = risk_validation.reconstruct_families("dev", dev_inputs)
    dev_result = evaluate_frozen_dev(
        frozen_winner, matrices[frozen_winner], dev_inputs, train_gains,
        train_costs, dev_gains, dev_increments, dev_costs,
        dev_families,
        train_results[frozen_winner],
    )
    decision = {
        "diagnostic_winner": frozen_winner,
        "train_representation_gate_passed": train_gate_passed,
        "candidate_adopted": False,
        "submission_default": "safe-margin",
        "reason": (
            "Train representation gate failed; downstream safety and champion gates were not opened."
            if not train_gate_passed
            else "Train gate passed, but integration requires unchanged safety and Dev champion validation."
        ),
        "downstream_gates": {
            "risk_validation_resamples": 5000,
            "headroom_tolerance": 0.005,
            "dev_weighted_champion_threshold": DEV_CHAMPION_THRESHOLD,
            "evaluated": False,
            "passed": False,
        },
    }
    report = {
        "report_type": REPORT_TYPE,
        "protocol": {
            "selection_data": "public Train only",
            "cv": "leave-one-reconstructed-source/task-family-out",
            "groups": list(risk_validation.FAMILY_LABELS),
            "family_to_fold": family_to_fold,
            "calibration_and_conformal_requirement": "group-disjoint; unchanged v2 path required only after Train adoption gate",
            "forbidden_runtime_features": ["source/task-family label", "episode_id", "outcome", "Dev outcome", "split identity", "row position"],
            "alpha_candidates": list(ALPHAS),
            "spend_goals": SPEND_GOALS,
            "gate_thresholds": {
                "correlation_floor_each_upgrade": TRAIN_CORRELATION_FLOOR,
                "positive_families_each_upgrade": MIN_POSITIVE_FAMILIES,
                "selected_gain_margin_over_current": SELECTED_GAIN_MARGIN,
                "per_family_selected_gain_floor": FAMILY_GAIN_FLOOR,
            },
        },
        "inputs": {
            "train_input_sha256": _sha256(train_input),
            "train_outcomes_sha256": _sha256(train_outcomes),
            "dev_input_sha256": _sha256(dev_input),
            "dev_outcomes_sha256": _sha256(dev_outcomes),
        },
        "train_representations": train_results,
        "frozen_dev_evaluation": dev_result,
        "decision": decision,
        "limitations": [
            "Nine public families are a small and heterogeneous group sample.",
            "Hash collisions and lexical overlap are only semantic proxies, not pretrained semantic understanding.",
            "Public Dev is one final diagnostic evaluation and is not evidence about the private evaluation set.",
        ],
    }
    _write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the family-disjoint representation audit")
    parser.add_argument("--train-input", type=Path, default=ROOT / "data/materialized/train/inputs.json")
    parser.add_argument("--train-outcomes", type=Path, default=ROOT / "data/train/outcomes.json")
    parser.add_argument("--dev-input", type=Path, default=ROOT / "data/materialized/dev/inputs.json")
    parser.add_argument("--dev-outcomes", type=Path, default=ROOT / "data/dev/outcomes.json")
    parser.add_argument("--report", type=Path, default=ROOT / "baselines/representation-audit-report.v1.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_audit(args.train_input, args.train_outcomes, args.dev_input, args.dev_outcomes, args.report)
    except (OSError, ValueError, risk_validation.RiskValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    decision = report["decision"]
    print(f"OK: winner={decision['diagnostic_winner']} adopted={decision['candidate_adopted']} default={decision['submission_default']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
