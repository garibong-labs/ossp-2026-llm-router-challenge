# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Train the risk-calibrated v2 router from public Train outcomes only.

Development-only tool (NumPy). The public Dev split is *not* read here: every
model choice, calibration and tier-plan value is decided from deterministic
out-of-fold (OOF) predictions on public Train. Dev is reserved for the
one-shot champion evaluation done by ``tools/risk_validation.py``.

What is fitted and recorded:

* direct upgrade heads – expected gain and probability of positive gain for
  ``ax31-light -> ax31`` and ``ax31 -> axk1-think`` – plus per-model log-cost
  heads, each with its own ridge strength selected by OOF error;
* OOF Platt vs isotonic probability calibration, judged by parity-split Brier
  score; the simpler Platt map wins ties;
* split-conformal per-model cost upper factors: the factor is the declared
  quantile of the even-half OOF log-cost residuals and the measured coverage
  comes from the disjoint odd half, so the recorded coverage is honest;
* a bounded candidate comparison (A: linear hash heads, B: compact boosted
  trees on dense structural features only, C: their blend), judged by OOF
  selected-set realized gain at the frozen spend goals;
* per-tier plans: guard grid chosen by OOF planner quality, then the target
  ratio binary-searched so the OOF realized Train ratio matches the frozen
  safe-margin spend goal (equal measured risk spend, not a budget expansion).

```console
PYTHONPATH=src python3 baselines/train_risk_calibrated.py \
  --input data/materialized/train/inputs.json \
  --outcomes data/train/outcomes.json \
  --artifact build/risk-calibrated/artifact.json \
  --report build/risk-calibrated/train-report.json
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by the CLI error path
    np = None

_BASELINES_DIRECTORY = str(Path(__file__).resolve().parent)
if _BASELINES_DIRECTORY not in sys.path:
    sys.path.insert(0, _BASELINES_DIRECTORY)

import hash_regex  # noqa: E402
import risk_calibrated  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    InputBatch,
    Outcome,
    OutcomeBatch,
    ProtocolError,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    policy_sha256,
)


DEFAULT_HASH_BINS = 256
DEFAULT_FOLDS = 5
DEFAULT_ALPHAS = (30.0, 100.0, 300.0, 1000.0, 3000.0)
DECLARED_COVERAGE = 0.85

#: Frozen Train realized-spend goals: the safe-margin policy's own measured
#: Train ratios. Matching them keeps the candidate's spend level – and
#: therefore its measured composition risk – comparable by construction.
SPEND_GOALS = {"fast": 1.098738, "balanced": 1.397907, "premium": 2.581280}

#: Hard ceilings for the planned target-ratio search. The planner spends its
#: budget in *risk-adjusted* (conformal upper) cost units, which overstate the
#: realized spend by roughly the upper factors, so these ceilings still keep
#: the realized ratio far below every hard cap; the plan-time clamp
#: ``min(target, cap)`` remains as the final defensive bound.
TARGET_SEARCH_CEILING = {"fast": 1.22, "balanced": 1.90, "premium": 4.00}

#: Deterministic guard grids per tier (upper-cost denominated).
GUARD_GRID = {
    "fast": {
        "min_probability": (0.0, 0.15, 0.30),
        "max_step_load": (3.0, 4.0, 6.0),
        "min_gain": (0.008,),
        "max_step_ratio": 8.0,
    },
    "balanced": {
        "min_probability": (0.0, 0.15, 0.30),
        "max_step_load": (6.0, 9.0, 12.0),
        "min_gain": (0.004,),
        "max_step_ratio": 12.0,
    },
    "premium": {
        "min_probability": (0.0, 0.10),
        "max_step_load": (12.0, 18.0),
        "min_gain": (0.0, 0.002),
        "max_step_ratio": 60.0,
    },
}
THINK_GRID = {
    "min_gain": (0.02, 0.05, 0.10),
    "max_step_load": (10.0, 16.0, 22.0),
    "min_probability": (0.0, 0.10),
    "max_step_ratio": 200.0,
    "budget_share": 0.65,
}

_STEPS = ("ax31", "axk1-think")


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError(
            "학습에는 NumPy가 필요합니다. baselines/requirements-train.txt를 "
            "설치해 주세요."
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outcome_cost(outcome: Outcome, policy: RoutingPolicy) -> float:
    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    cost = (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )
    result = float(cost)
    if not math.isfinite(result) or result <= 0:
        raise ProtocolError("학습 outcome의 모델 비용은 0보다 커야 합니다.")
    return result


def _training_tables(
    inputs: InputBatch,
    outcomes: OutcomeBatch,
    policy: RoutingPolicy,
    hash_bins: int,
) -> Tuple[Any, Any, Any]:
    """Return the feature matrix plus realized score and cost tables."""

    _require_numpy()
    if inputs.schema_version != outcomes.schema_version:
        raise ProtocolError("Train 입력과 outcome의 schema_version이 다릅니다.")
    if inputs.challenge_id != outcomes.challenge_id or inputs.split != outcomes.split:
        raise ProtocolError("Train 입력과 outcome의 실행 메타데이터가 다릅니다.")
    outcome_index = {
        (outcome.episode_id, outcome.model_id): outcome
        for outcome in outcomes.outcomes
    }
    expected = {
        (episode.episode_id, model_id)
        for episode in inputs.episodes
        for model_id in MODEL_IDS
    }
    if set(outcome_index) != expected:
        raise ProtocolError("Train outcome 행렬이 입력과 모델 전체를 포함하지 않습니다.")
    matrix = np.asarray(
        [
            hash_regex.raw_feature_vector(episode, hash_bins)
            for episode in inputs.episodes
        ],
        dtype=np.float64,
    )
    scores = np.asarray(
        [
            [
                float(outcome_index[(episode.episode_id, model_id)].score)
                for model_id in MODEL_IDS
            ]
            for episode in inputs.episodes
        ],
        dtype=np.float64,
    )
    costs = np.asarray(
        [
            [
                _outcome_cost(
                    outcome_index[(episode.episode_id, model_id)], policy
                )
                for model_id in MODEL_IDS
            ]
            for episode in inputs.episodes
        ],
        dtype=np.float64,
    )
    return matrix, scores, costs


def _fit_ridge(matrix: Any, targets: Any, alpha: float) -> Tuple[Any, Any, Any, Any]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = targets.mean(axis=0)
    centered = targets - intercept
    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + alpha * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
    else:
        system = standardized.T @ standardized + alpha * np.eye(columns)
        coefficients = np.linalg.solve(system, standardized.T @ centered)
    return mean, scale, intercept, coefficients


def _predict_ridge(matrix, mean, scale, intercept, coefficients):
    return (matrix - mean) / scale @ coefficients + intercept


def _oof_predictions(matrix, targets, *, folds: int, alpha: float):
    rows = matrix.shape[0]
    predictions = np.empty_like(targets)
    fold_ids = np.arange(rows) % folds
    for fold in range(folds):
        validation = fold_ids == fold
        training = ~validation
        mean, scale, intercept, coefficients = _fit_ridge(
            matrix[training], targets[training], alpha
        )
        predictions[validation] = _predict_ridge(
            matrix[validation], mean, scale, intercept, coefficients
        )
    return predictions


def _select_alpha_by(
    matrix: Any,
    targets: Any,
    *,
    folds: int,
    candidates: Sequence[float],
    metric,
) -> Tuple[float, Any, Mapping[str, float]]:
    """Pick the ridge strength whose OOF predictions minimize ``metric``."""

    best = None
    diagnostics: Dict[str, float] = {}
    for alpha in candidates:
        predictions = _oof_predictions(matrix, targets, folds=folds, alpha=alpha)
        objective = float(metric(predictions, targets))
        diagnostics[format(alpha, ".12g")] = objective
        rank = (objective, alpha)
        if best is None or rank < best[0]:
            best = (rank, alpha, predictions)
    assert best is not None
    return best[1], best[2], diagnostics


# --------------------------------------------------------------------------
# Probability calibration


def _fit_platt(raw: Any, labels: Any) -> risk_calibrated.PlattCalibration:
    """Deterministic Newton fit of P(y=1 | s) = sigmoid(a*s + b)."""

    a, b = 1.0, 0.0
    for _iteration in range(100):
        z = np.clip(a * raw + b, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.maximum(p * (1.0 - p), 1e-12)
        g_a = np.sum((p - labels) * raw)
        g_b = np.sum(p - labels)
        h_aa = np.sum(w * raw * raw) + 1e-9
        h_ab = np.sum(w * raw)
        h_bb = np.sum(w) + 1e-9
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-18:
            break
        step_a = (h_bb * g_a - h_ab * g_b) / det
        step_b = (h_aa * g_b - h_ab * g_a) / det
        a -= step_a
        b -= step_b
        if abs(step_a) < 1e-12 and abs(step_b) < 1e-12:
            break
    return risk_calibrated.PlattCalibration(scale=max(a, 1e-6), offset=b)


def _fit_isotonic(raw: Any, labels: Any) -> risk_calibrated.IsotonicCalibration:
    """Pool-adjacent-violators on the raw-score ordering."""

    order = np.lexsort((np.arange(len(raw)), raw))
    xs = raw[order]
    ys = labels[order]
    blocks: List[List[float]] = []  # [sum, count, x_start]
    for x, y in zip(xs, ys):
        blocks.append([float(y), 1.0, float(x)])
        while len(blocks) > 1 and (
            blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
        ):
            last = blocks.pop()
            blocks[-1][0] += last[0]
            blocks[-1][1] += last[1]
    thresholds = tuple(block[2] for block in blocks)
    values = tuple(block[0] / block[1] for block in blocks)
    return risk_calibrated.IsotonicCalibration(
        thresholds=thresholds, values=values
    )


def _brier(calibration, raw: Any, labels: Any) -> float:
    predicted = np.asarray([calibration.apply(float(value)) for value in raw])
    return float(np.mean((predicted - labels) ** 2))


def _choose_calibration(
    raw: Any, labels: Any
) -> Tuple[Any, Mapping[str, float]]:
    """Parity-split comparison; the simpler Platt map wins ties."""

    even = np.arange(len(raw)) % 2 == 0
    scores = {"platt": 0.0, "isotonic": 0.0}
    for half, other in ((even, ~even), (~even, even)):
        platt = _fit_platt(raw[half], labels[half])
        isotonic = _fit_isotonic(raw[half], labels[half])
        scores["platt"] += _brier(platt, raw[other], labels[other]) / 2.0
        scores["isotonic"] += _brier(isotonic, raw[other], labels[other]) / 2.0
    if scores["isotonic"] < scores["platt"] - 1e-4:
        chosen = _fit_isotonic(raw, labels)
        method = "isotonic"
    else:
        chosen = _fit_platt(raw, labels)
        method = "platt"
    return chosen, {"method": method, **scores}


# --------------------------------------------------------------------------
# Candidate B: compact boosted depth-2 trees on dense structural features


def _fit_boosted_trees(
    X: Any, y: Any, *, rounds: int = 120, learning_rate: float = 0.1,
    min_leaf: int = 40,
) -> List[Any]:
    thresholds = [
        np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, 19)))
        for j in range(X.shape[1])
    ]
    prediction = np.full(len(y), float(y.mean()))
    trees: List[Any] = [float(y.mean())]

    def best_split(idx, residual):
        best = None
        for j in range(X.shape[1]):
            xs = X[idx, j]
            for t in thresholds[j]:
                left = idx[xs <= t]
                right = idx[xs > t]
                if len(left) < min_leaf or len(right) < min_leaf:
                    continue
                gain = (
                    residual[left].sum() ** 2 / len(left)
                    + residual[right].sum() ** 2 / len(right)
                )
                key = (gain, -j, -t)
                if best is None or key > best[0]:
                    best = (key, j, t, left, right)
        return best

    all_idx = np.arange(len(y))
    for _round in range(rounds):
        residual = y - prediction
        root = best_split(all_idx, residual)
        if root is None:
            break
        _, j, t, left, right = root
        children = {}
        for side, idx in (("le", left), ("gt", right)):
            sub = best_split(idx, residual)
            if sub is None:
                children[side] = ("leaf", float(residual[idx].mean()))
            else:
                _, j2, t2, l2, r2 = sub
                children[side] = (
                    "split",
                    int(j2),
                    float(t2),
                    float(residual[l2].mean()),
                    float(residual[r2].mean()),
                )
        tree = (int(j), float(t), children)
        prediction += learning_rate * _apply_tree(X, tree)
        trees.append((tree, learning_rate))
    return trees


def _apply_tree(X: Any, tree: Any) -> Any:
    j, t, children = tree
    out = np.empty(len(X))
    left_mask = X[:, j] <= t
    for side, mask in (("le", left_mask), ("gt", ~left_mask)):
        node = children[side]
        if node[0] == "leaf":
            out[mask] = node[1]
        else:
            _, j2, t2, vl, vr = node
            sub = X[mask, j2] <= t2
            values = np.where(sub, vl, vr)
            out[mask] = values
    return out


def _predict_boosted_trees(X: Any, model: List[Any]) -> Any:
    out = np.full(len(X), model[0])
    for tree, lr in model[1:]:
        out += lr * _apply_tree(X, tree)
    return out


def _oof_boosted_trees(X: Any, y: Any, folds: int) -> Any:
    out = np.empty_like(y)
    fold_ids = np.arange(len(y)) % folds
    for fold in range(folds):
        validation = fold_ids == fold
        model = _fit_boosted_trees(X[~validation], y[~validation])
        out[validation] = _predict_boosted_trees(X[validation], model)
    return out


def _selected_set_gain(
    pred_gain: Any, pred_increment: Any, real_increment: Any, real_gain: Any,
    light_total: float, target_ratio: float,
) -> float:
    """Realized mean gain of the OOF-ranked selection at a fixed spend."""

    efficiency = pred_gain / np.maximum(pred_increment, 1e-12)
    order = np.argsort(-efficiency)
    budget = (target_ratio - 1.0) * light_total
    spent = 0.0
    total_gain = 0.0
    for index in order:
        if pred_gain[index] <= 0:
            continue
        if spent + real_increment[index] <= budget:
            spent += real_increment[index]
            total_gain += real_gain[index]
    return total_gain / len(pred_gain)


# --------------------------------------------------------------------------
# Tier-plan tuning on OOF predictions


def _prediction_objects(
    gains1: Any,
    gains2: Any,
    probs1: Any,
    probs2: Any,
    cost_rows: Any,
    upper_factors: Mapping[str, float],
    signatures: Sequence[Tuple[int, ...]],
    policy: RoutingPolicy,
) -> Tuple[Any, ...]:
    predictions = []
    for index in range(len(gains1)):
        raw_costs = {
            model_id: float(cost_rows[index][j])
            for j, model_id in enumerate(MODEL_IDS)
        }
        costs = risk_calibrated._floored_monotone(raw_costs, policy)
        upper = risk_calibrated._floored_monotone(
            {
                model_id: costs[model_id] * upper_factors[model_id]
                for model_id in MODEL_IDS
            },
            policy,
        )
        predictions.append(
            risk_calibrated.EpisodePredictionV2(
                gains={
                    "ax31": float(np.clip(gains1[index], -1.0, 1.0)),
                    "axk1-think": float(np.clip(gains2[index], -1.0, 1.0)),
                },
                probabilities={
                    "ax31": float(np.clip(probs1[index], 0.0, 1.0)),
                    "axk1-think": float(np.clip(probs2[index], 0.0, 1.0)),
                },
                costs=costs,
                upper_costs=upper,
                signature=tuple(signatures[index]),
            )
        )
    return tuple(predictions)


def _plan_metrics(
    predictions: Sequence[Any],
    policy: RoutingPolicy,
    tier: str,
    plan: risk_calibrated.TierPlanV2,
    real_scores: Any,
    real_costs: Any,
) -> Tuple[float, float]:
    plans = {tier: plan}
    selected, _ratio, _stages = risk_calibrated.plan_selection(
        predictions, policy, tier, plans
    )
    model_index = {model_id: j for j, model_id in enumerate(MODEL_IDS)}
    chosen = np.asarray([model_index[model_id] for model_id in selected])
    quality = float(
        real_scores[np.arange(len(chosen)), chosen].mean()
    )
    total = float(real_costs[np.arange(len(chosen)), chosen].sum())
    ratio = total / float(real_costs[:, 0].sum())
    return quality, ratio


def _tune_tier_plan(
    tier: str,
    predictions_for,
    policy: RoutingPolicy,
    real_scores: Any,
    real_costs: Any,
) -> Tuple[risk_calibrated.TierPlanV2, List[Mapping[str, Any]]]:
    """Grid the guards, then binary-search the target to the spend goal."""

    grid = GUARD_GRID[tier]
    goal = SPEND_GOALS[tier]
    trace: List[Mapping[str, Any]] = []
    best = None
    think_options: Sequence[Tuple[float, float, float]] = [(1.0, 0.0, 0.0)]
    if tier == "premium":
        think_options = [
            (think_gain, think_load, think_prob)
            for think_gain in THINK_GRID["min_gain"]
            for think_load in THINK_GRID["max_step_load"]
            for think_prob in THINK_GRID["min_probability"]
        ]
    for min_gain in grid["min_gain"]:
      for min_probability in grid["min_probability"]:
        for max_step_load in grid["max_step_load"]:
            for think_gain, think_load, think_prob in think_options:
                def make_plan(target: float) -> risk_calibrated.TierPlanV2:
                    return risk_calibrated.TierPlanV2(
                        target_ratio=target,
                        min_gain=min_gain,
                        min_probability=min_probability,
                        max_step_ratio=grid["max_step_ratio"],
                        max_step_load=max_step_load,
                        allow_think=tier == "premium",
                        think_min_gain=think_gain,
                        think_min_probability=think_prob,
                        think_max_step_ratio=THINK_GRID["max_step_ratio"],
                        think_max_step_load=think_load,
                        think_budget_share=THINK_GRID["budget_share"],
                    )

                low = 1.0
                high = TARGET_SEARCH_CEILING[tier]
                for _step in range(20):
                    middle = (low + high) / 2.0
                    _quality, ratio = _plan_metrics(
                        predictions_for,
                        policy,
                        tier,
                        make_plan(middle),
                        real_scores,
                        real_costs,
                    )
                    if ratio <= goal:
                        low = middle
                    else:
                        high = middle
                target = low
                quality, ratio = _plan_metrics(
                    predictions_for,
                    policy,
                    tier,
                    make_plan(target),
                    real_scores,
                    real_costs,
                )
                entry = {
                    "min_gain": min_gain,
                    "min_probability": min_probability,
                    "max_step_load": max_step_load,
                    "think_min_gain": think_gain,
                    "think_max_step_load": think_load,
                    "think_min_probability": think_prob,
                    "target_ratio": target,
                    "oof_realized_ratio": ratio,
                    "oof_quality": quality,
                }
                trace.append(entry)
                key = (quality, -ratio, -min_probability)
                if best is None or key > best[0]:
                    best = (key, make_plan(target))
    assert best is not None
    return best[1], trace


# --------------------------------------------------------------------------


def _head_dict(intercept: float, coefficients: Any) -> Mapping[str, Any]:
    return {
        "intercept": float(intercept),
        "coefficients": [float(value) for value in coefficients],
    }


def _calibration_dict(calibration: Any) -> Mapping[str, Any]:
    if isinstance(calibration, risk_calibrated.PlattCalibration):
        return {
            "method": "platt",
            "scale": calibration.scale,
            "offset": calibration.offset,
        }
    return {
        "method": "isotonic",
        "thresholds": list(calibration.thresholds),
        "values": list(calibration.values),
    }


def _plan_dict(plan: risk_calibrated.TierPlanV2) -> Mapping[str, Any]:
    return {field: getattr(plan, field) for field in plan.__dataclass_fields__}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o644)
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def train(
    *,
    input_path: Path,
    outcomes_path: Path,
    artifact_path: Path,
    report_path: Path,
    policy: RoutingPolicy,
    hash_bins: int = DEFAULT_HASH_BINS,
    folds: int = DEFAULT_FOLDS,
    alpha_candidates: Sequence[float] = DEFAULT_ALPHAS,
) -> Mapping[str, Any]:
    _require_numpy()
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("Train 입력과 정책의 schema_version이 다릅니다.")
    if folds < 2:
        raise ValueError("OOF fold 수는 2 이상이어야 합니다.")
    matrix, scores, costs = _training_tables(inputs, outcomes, policy, hash_bins)
    signatures = [
        risk_calibrated.content_signature(row) for row in matrix.tolist()
    ]

    gain1 = scores[:, 1] - scores[:, 0]
    gain2 = scores[:, 2] - scores[:, 1]
    label1 = (gain1 > 0).astype(np.float64)
    label2 = (gain2 > 0).astype(np.float64)
    log_costs = np.log(costs)

    # --- per-group ridge strengths ------------------------------------
    def mse(predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def brier(predictions, targets):
        return np.mean((np.clip(predictions, 0.0, 1.0) - targets) ** 2)

    gains = np.column_stack([gain1, gain2])
    labels = np.column_stack([label1, label2])
    alpha_cost, oof_log_costs, cost_diag = _select_alpha_by(
        matrix, log_costs, folds=folds, candidates=alpha_candidates, metric=mse
    )

    # The gain heads exist to *rank* upgrades, so their ridge strength is
    # selected by the routing-relevant OOF metric: realized gain of the
    # ranked selection at the frozen spend goals (not by regression MSE,
    # which degenerates into maximal shrinkage on a noisy target).
    light_total_for_metric = float(costs[:, 0].sum())
    real_inc1_for_metric = costs[:, 1] - costs[:, 0]
    pred_inc1_for_metric = np.maximum(
        np.exp(oof_log_costs[:, 1]) - np.exp(oof_log_costs[:, 0]),
        np.exp(oof_log_costs[:, 0]) * 0.5,
    )

    def negative_selected_set_gain(predictions, targets):
        return -sum(
            _selected_set_gain(
                predictions[:, 0],
                pred_inc1_for_metric,
                real_inc1_for_metric,
                targets[:, 0],
                light_total_for_metric,
                SPEND_GOALS[tier],
            )
            for tier in ("fast", "balanced")
        )

    alpha_gain, oof_gains, gain_diag = _select_alpha_by(
        matrix,
        gains,
        folds=folds,
        candidates=alpha_candidates,
        metric=negative_selected_set_gain,
    )
    alpha_prob, oof_probs_raw, prob_diag = _select_alpha_by(
        matrix, labels, folds=folds, candidates=alpha_candidates, metric=brier
    )

    # --- probability calibration --------------------------------------
    calibrations = {}
    calibration_diag = {}
    for j, step in enumerate(_STEPS):
        calibration, diagnostics = _choose_calibration(
            oof_probs_raw[:, j], labels[:, j]
        )
        calibrations[step] = calibration
        calibration_diag[step] = diagnostics

    # --- split-conformal cost upper factors ---------------------------
    residuals = log_costs - oof_log_costs
    even = np.arange(len(matrix)) % 2 == 0
    upper_factors: Dict[str, float] = {}
    measured_coverage: Dict[str, float] = {}
    for j, model_id in enumerate(MODEL_IDS):
        calibrating = np.sort(residuals[even, j])
        rank = min(
            len(calibrating) - 1,
            max(0, math.ceil(DECLARED_COVERAGE * (len(calibrating) + 1)) - 1),
        )
        factor = float(np.exp(calibrating[rank]))
        factor = max(1.0, factor)
        coverage = float(np.mean(residuals[~even, j] <= math.log(factor)))
        upper_factors[model_id] = factor
        measured_coverage[model_id] = coverage

    # --- bounded candidate comparison (A vs B vs C) --------------------
    light_total = float(costs[:, 0].sum())
    real_inc1 = costs[:, 1] - costs[:, 0]
    dense = matrix[:, : len(hash_regex.DENSE_FEATURE_NAMES)]
    oof_tree_gain1 = _oof_boosted_trees(dense, gain1, folds)
    pred_inc1 = np.maximum(
        np.exp(oof_log_costs[:, 1]) - np.exp(oof_log_costs[:, 0]),
        np.exp(oof_log_costs[:, 0]) * 0.5,
    )
    candidate_metrics: Dict[str, Any] = {}
    for name, oof_gain in (
        ("A-linear-hash", oof_gains[:, 0]),
        ("B-dense-trees", oof_tree_gain1),
        ("C-blend", 0.5 * (oof_gains[:, 0] + oof_tree_gain1)),
    ):
        candidate_metrics[name] = {
            "oof_mse": float(mse(oof_gain, gain1)),
            "oof_correlation": float(np.corrcoef(oof_gain, gain1)[0, 1]),
            "selected_set_gain": {
                tier: _selected_set_gain(
                    oof_gain,
                    pred_inc1,
                    real_inc1,
                    gain1,
                    light_total,
                    SPEND_GOALS[tier],
                )
                for tier in ("fast", "balanced")
            },
        }
    a_scores = candidate_metrics["A-linear-hash"]["selected_set_gain"]
    chosen_candidate = "A-linear-hash"
    for name in ("B-dense-trees", "C-blend"):
        other = candidate_metrics[name]["selected_set_gain"]
        if all(other[tier] >= a_scores[tier] + 0.002 for tier in a_scores):
            chosen_candidate = name
    if chosen_candidate != "A-linear-hash":
        raise RuntimeError(
            "비선형 후보가 우세합니다: 직렬화 경로를 추가로 검토해야 합니다: "
            f"{chosen_candidate}"
        )

    # --- tier plans from OOF planner quality ---------------------------
    oof_probs_cal = np.column_stack(
        [
            np.asarray(
                [
                    calibrations[step].apply(float(value))
                    for value in oof_probs_raw[:, j]
                ]
            )
            for j, step in enumerate(_STEPS)
        ]
    )
    oof_predictions = _prediction_objects(
        oof_gains[:, 0],
        oof_gains[:, 1],
        oof_probs_cal[:, 0],
        oof_probs_cal[:, 1],
        np.exp(oof_log_costs),
        upper_factors,
        signatures,
        policy,
    )
    tier_plans: Dict[str, risk_calibrated.TierPlanV2] = {}
    tuning_trace: Dict[str, Any] = {}
    for tier in TIERS:
        plan, trace = _tune_tier_plan(
            tier, oof_predictions, policy, scores, costs
        )
        tier_plans[tier] = plan
        tuning_trace[tier] = trace

    # --- final full-train heads ----------------------------------------
    mean_g, scale_g, icept_g, coef_g = _fit_ridge(matrix, gains, alpha_gain)
    mean_p, scale_p, icept_p, coef_p = _fit_ridge(matrix, labels, alpha_prob)
    mean_c, scale_c, icept_c, coef_c = _fit_ridge(matrix, log_costs, alpha_cost)
    # All heads share the artifact's single standardization block; refit the
    # gain/probability/cost heads against the cost head's statistics so one
    # (mean, scale) pair serializes. The statistics are identical because they
    # come from the same matrix; assert instead of trusting that silently.
    assert np.allclose(mean_g, mean_c) and np.allclose(scale_g, scale_c)
    assert np.allclose(mean_p, mean_c) and np.allclose(scale_p, scale_c)

    training_summary = {
        "num_episodes": len(inputs.episodes),
        "folds": folds,
        "hash_bins": hash_bins,
        "alpha_gain": alpha_gain,
        "alpha_probability": alpha_prob,
        "alpha_cost": alpha_cost,
        "declared_coverage": DECLARED_COVERAGE,
        "spend_goals": dict(SPEND_GOALS),
        "chosen_candidate": chosen_candidate,
        "input_sha256": _file_sha256(input_path),
        "outcomes_sha256": _file_sha256(outcomes_path),
        "optimizer": "numpy-ridge-oof-riskcal-v2",
    }
    artifact_value = {
        "artifact_type": risk_calibrated.ARTIFACT_TYPE,
        "schema_version": 1,
        "feature_version": hash_regex.FEATURE_VERSION,
        "hash_algorithm": "fnv1a64-signed-word-1-2",
        "hash_bins": hash_bins,
        "dense_feature_names": list(hash_regex.DENSE_FEATURE_NAMES),
        "model_ids": list(MODEL_IDS),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "feature_mean": [float(value) for value in mean_c],
        "feature_scale": [float(value) for value in scale_c],
        "gain_heads": {
            step: _head_dict(icept_g[j], coef_g[:, j])
            for j, step in enumerate(_STEPS)
        },
        "gain_probability_heads": {
            step: _head_dict(icept_p[j], coef_p[:, j])
            for j, step in enumerate(_STEPS)
        },
        "gain_probability_calibration": {
            step: _calibration_dict(calibrations[step]) for step in _STEPS
        },
        "log_cost_heads": {
            model_id: _head_dict(icept_c[j], coef_c[:, j])
            for j, model_id in enumerate(MODEL_IDS)
        },
        "cost_upper": {
            "declared_coverage": DECLARED_COVERAGE,
            "factors": upper_factors,
            "measured_coverage": measured_coverage,
        },
        "tier_plans": {
            tier: _plan_dict(tier_plans[tier]) for tier in TIERS
        },
        "training_summary": training_summary,
    }
    artifact = risk_calibrated.parse_artifact(artifact_value)

    # --- target recalibration on the shipped heads ---------------------
    # The guard grid above is chosen from OOF quality so it cannot overfit
    # the shipped heads, but the *spend* calibration must describe the heads
    # that actually run: the full-train fit is sharper than its OOF preview,
    # so each tier's target ratio is re-searched with the fitted predictions
    # until the realized Train ratio matches the frozen spend goal again.
    from dataclasses import replace as _replace

    fitted_predictions = risk_calibrated.predict_batch(
        inputs.episodes, artifact, policy
    )
    recalibrated_targets: Dict[str, Mapping[str, float]] = {}
    for tier in TIERS:
        plan = tier_plans[tier]
        low = 1.0
        high = TARGET_SEARCH_CEILING[tier]
        for _step in range(20):
            middle = (low + high) / 2.0
            _quality, ratio = _plan_metrics(
                fitted_predictions,
                policy,
                tier,
                _replace(plan, target_ratio=middle),
                scores,
                costs,
            )
            if ratio <= SPEND_GOALS[tier]:
                low = middle
            else:
                high = middle
        recalibrated_targets[tier] = {
            "oof_target_ratio": plan.target_ratio,
            "fitted_target_ratio": low,
        }
        tier_plans[tier] = _replace(plan, target_ratio=low)
    artifact_value["tier_plans"] = {
        tier: _plan_dict(tier_plans[tier]) for tier in TIERS
    }
    artifact = risk_calibrated.parse_artifact(artifact_value)
    self_check = {}
    for tier in TIERS:
        selected, planned_ratio, _stages = risk_calibrated.plan_selection(
            fitted_predictions, policy, tier, artifact.tier_plans
        )
        model_index = {model_id: j for j, model_id in enumerate(MODEL_IDS)}
        chosen = np.asarray([model_index[model_id] for model_id in selected])
        rows = np.arange(len(chosen))
        self_check[tier] = {
            "planned_ratio": planned_ratio,
            "realized_ratio": float(costs[rows, chosen].sum() / costs[:, 0].sum()),
            "realized_quality": float(scores[rows, chosen].mean()),
            "model_counts": {
                model_id: int((chosen == j).sum())
                for j, model_id in enumerate(MODEL_IDS)
            },
        }

    report = {
        "report_type": "ossp-risk-calibrated-training-v2",
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "training_summary": training_summary,
        "alpha_objectives": {
            "gain": gain_diag,
            "probability": prob_diag,
            "log_cost": cost_diag,
        },
        "probability_calibration": calibration_diag,
        "cost_upper": {
            "declared_coverage": DECLARED_COVERAGE,
            "factors": upper_factors,
            "measured_coverage": measured_coverage,
        },
        "candidate_comparison": candidate_metrics,
        "tier_plan_tuning": tuning_trace,
        "target_recalibration": recalibrated_targets,
        "tier_plans": {tier: _plan_dict(tier_plans[tier]) for tier in TIERS},
        "fitted_train_self_check": self_check,
    }
    _write_json_atomic(artifact_path, artifact_value)
    _write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="공개 Train으로 위험 보정 v2 라우터를 학습합니다."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--hash-bins", type=int, default=DEFAULT_HASH_BINS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
        report = train(
            input_path=args.input,
            outcomes_path=args.outcomes,
            artifact_path=args.artifact,
            report_path=args.report,
            policy=policy,
            hash_bins=args.hash_bins,
            folds=args.folds,
        )
    except (OSError, ProtocolError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    fast = report["fitted_train_self_check"]["fast"]
    print(
        "OK: risk-calibrated v2 artifact를 생성했습니다 "
        f"(Train fast self-check {fast['realized_quality']:.6f}@"
        f"{fast['realized_ratio']:.4f})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
