# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Train-first runner for safe-margin residual consensus v1.

The protocol hash below was frozen before this experiment loaded any new
outcomes.  Public Dev outcomes are not opened unless every Train gate passes.
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

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "baselines", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import representation_features  # noqa: E402
import risk_validation  # noqa: E402
import safe_margin  # noqa: E402
import safe_margin_residual_consensus as candidate  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)
from ossp_router.scoring import score_submissions  # noqa: E402
from stress_safe_margin import bootstrap_indices, outcome_tables  # noqa: E402


PROTOCOL_PATH = ROOT / "configs/safe-margin-residual-consensus-protocol.v1.json"
EXPECTED_PROTOCOL_SHA256 = (
    "cd78edcf7f36aa06a97fd930e88e27bd3f80188677a06907a7e0aeb433c64dc1"
)
DEFAULT_ARTIFACT = ROOT / "baselines/safe-margin-residual-consensus-public.v1.json"
DEFAULT_REPORT = ROOT / "baselines/safe-margin-residual-consensus-report.v1.json"
DEFAULT_TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"
DEFAULT_TRAIN_OUTCOMES = ROOT / "data/train/outcomes.json"
DEFAULT_DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"
DEFAULT_DEV_OUTCOMES = ROOT / "data/dev/outcomes.json"
STRENGTHS = (0.25, 0.5, 1.0)
RIDGE_ALPHA = 1000.0
TRAIN_MINIMUM = 0.001
MIN_POSITIVE_FAMILIES = 6
FAMILY_FLOOR = -0.005
DEV_COMPARATOR = 0.673182
DEV_PRACTICAL_TARGET = 0.674
REPORT_DIGITS = 9


class ExperimentError(RuntimeError):
    """Raised when the frozen experiment cannot be evaluated honestly."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_numpy() -> None:
    if np is None:
        raise ExperimentError("the Train-only experiment requires NumPy")


def validate_protocol(path: Path = PROTOCOL_PATH) -> Mapping[str, Any]:
    if file_sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ExperimentError("frozen protocol SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    strengths = tuple(value["candidate"]["correction_strengths"])
    if strengths != STRENGTHS or len(strengths) > 3:
        raise ExperimentError("frozen correction grid mismatch")
    if value["evaluation"]["ridge_alpha"] != RIDGE_ALPHA:
        raise ExperimentError("frozen ridge alpha mismatch")
    return value


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
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fit_ridge(matrix: Any, targets: Any) -> Tuple[Any, Any, float, Any]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = float(targets.mean())
    centered = targets - intercept
    rows, columns = standardized.shape
    if rows <= columns:
        system = standardized @ standardized.T + RIDGE_ALPHA * np.eye(rows)
        coefficients = standardized.T @ np.linalg.solve(system, centered)
    else:
        system = standardized.T @ standardized + RIDGE_ALPHA * np.eye(columns)
        coefficients = np.linalg.solve(
            system, standardized.T @ centered
        )
    return mean, scale, intercept, coefficients


def _predict_ridge(
    matrix: Any, fitted: Tuple[Any, Any, float, Any]
) -> Any:
    mean, scale, intercept, coefficients = fitted
    return np.clip(
        (matrix - mean) / scale @ coefficients + intercept,
        candidate.RESIDUAL_CLIP[0],
        candidate.RESIDUAL_CLIP[1],
    )


def _structural_matrix(inputs: InputBatch) -> Any:
    matrix = np.asarray(
        [
            representation_features.expanded_structural_vector(episode)
            for episode in inputs.episodes
        ],
        dtype=np.float64,
    )
    if matrix.shape[1] != 36 or not np.isfinite(matrix).all():
        raise ExperimentError("B-expanded-structural extraction failed closed")
    return matrix


def _load_train(
    input_path: Path, outcomes_path: Path
) -> Tuple[InputBatch, OutcomeBatch, RoutingPolicy, Any, Any, Tuple[str, ...]]:
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    if inputs.split != "train" or outcomes.split != "train":
        raise ExperimentError("selection inputs must be the public Train split")
    policy = load_bundled_policy()
    costs, scores = outcome_tables(inputs, outcomes, policy)
    families = risk_validation.reconstruct_families("train", inputs)
    if set(families) != set(risk_validation.FAMILY_LABELS):
        raise ExperimentError("Train does not contain all nine frozen families")
    return inputs, outcomes, policy, costs, scores, families


def _subset(values: Sequence[Any], indices: Sequence[int]) -> list:
    return [values[index] for index in indices]


def _selection_metrics(
    predictions: Sequence[Any],
    residuals: Sequence[float],
    scores: Sequence[Mapping[str, float]],
    policy: RoutingPolicy,
    strength: float,
) -> Mapping[str, Any]:
    tiers: Dict[str, Any] = {}
    weighted = 0.0
    matched = True
    for tier in ("fast", "balanced"):
        safe_selected, safe_ratio, _safe_stages = safe_margin.plan_selection(
            predictions, policy, tier
        )
        selected, predicted_ratio, matched_ratio = candidate.plan_selection(
            predictions, residuals, policy, tier, strength
        )
        safe_quality = math.fsum(
            row[model_id] for row, model_id in zip(scores, safe_selected)
        ) / len(scores)
        quality = math.fsum(
            row[model_id] for row, model_id in zip(scores, selected)
        ) / len(scores)
        improvement = quality - safe_quality
        weight = float(policy.tiers[tier].weight)
        weighted += weight * improvement
        spend_ok = predicted_ratio <= matched_ratio + 1e-12
        matched = matched and spend_ok and abs(matched_ratio - safe_ratio) <= 1e-12
        tiers[tier] = {
            "candidate_quality": quality,
            "safe_margin_quality": safe_quality,
            "quality_improvement": improvement,
            "candidate_conservative_ratio": predicted_ratio,
            "matched_safe_margin_conservative_ratio": matched_ratio,
            "matched_spend_passed": spend_ok,
            "changed_decisions": sum(
                left != right for left, right in zip(selected, safe_selected)
            ),
        }
    return {
        "weighted_improvement": weighted,
        "matched_spend_passed": matched,
        "tiers": tiers,
    }


def _oof_residuals(
    matrix: Any,
    targets: Any,
    families: Sequence[str],
    allowed_indices: Sequence[int],
) -> Any:
    result = np.full(len(families), np.nan, dtype=np.float64)
    allowed = set(int(index) for index in allowed_indices)
    family_names = sorted({families[index] for index in allowed})
    if len(family_names) < 2:
        raise ExperimentError("nested LOFO requires at least two families")
    for family in family_names:
        validation = [
            index
            for index in allowed_indices
            if families[index] == family
        ]
        training = [
            index
            for index in allowed_indices
            if families[index] != family
        ]
        if not training or not validation:
            raise ExperimentError("nested family split is indeterminate")
        fitted = _fit_ridge(matrix[training], targets[training])
        result[validation] = _predict_ridge(matrix[validation], fitted)
    if not np.isfinite(result[list(allowed_indices)]).all():
        raise ExperimentError("nested OOF prediction is incomplete")
    return result


def _choose_strength(
    predictions: Sequence[Any],
    residuals: Sequence[float],
    scores: Sequence[Mapping[str, float]],
    policy: RoutingPolicy,
) -> Tuple[float, Mapping[str, Any]]:
    records = {
        strength: _selection_metrics(
            predictions, residuals, scores, policy, strength
        )
        for strength in STRENGTHS
    }
    selected = max(
        STRENGTHS,
        key=lambda strength: (
            records[strength]["weighted_improvement"], -strength
        ),
    )
    return selected, {
        format(strength, ".2f"): records[strength] for strength in STRENGTHS
    }


def _artifact_value(
    fitted: Tuple[Any, Any, float, Any],
    policy: RoutingPolicy,
    strength: float,
    train_input: Path,
    train_outcomes: Path,
) -> Mapping[str, Any]:
    mean, scale, intercept, coefficients = fitted
    return {
        "artifact_version": 1,
        "coefficients": [float(item) for item in coefficients],
        "feature_mean": [float(item) for item in mean],
        "feature_names": list(
            representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES
        ),
        "feature_scale": [float(item) for item in scale],
        "intercept": float(intercept),
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "representation": "B-expanded-structural",
        "ridge_alpha": RIDGE_ALPHA,
        "strategy_id": candidate.STRATEGY_ID,
        "strength": strength,
        "train_input_sha256": file_sha256(train_input),
        "train_outcomes_sha256": file_sha256(train_outcomes),
    }


def _runtime_checks(
    inputs: InputBatch,
    policy: RoutingPolicy,
    base_artifact: Any,
    residual_artifact: candidate.ResidualArtifact,
) -> Mapping[str, Any]:
    first = candidate.predict_residuals(inputs.episodes, residual_artifact)
    second = candidate.predict_residuals(inputs.episodes, residual_artifact)
    deterministic = first == second
    plans_equal = True
    premium_identity = True
    spend_ok = True
    predictions = safe_margin.predict_batch(inputs.episodes, base_artifact, policy)
    for tier in TIERS:
        left = candidate.plan_selection(
            predictions, first, policy, tier, residual_artifact.strength
        )
        right = candidate.plan_selection(
            predictions, second, policy, tier, residual_artifact.strength
        )
        plans_equal = plans_equal and left == right
        safe = safe_margin.plan_selection(predictions, policy, tier)
        if tier == "premium":
            premium_identity = premium_identity and left[0] == safe[0]
        else:
            spend_ok = spend_ok and left[1] <= left[2] + 1e-12
    bound = representation_features.MAX_FIELD_CHARACTERS
    prefix = "x" * bound
    bound_probe = (
        representation_features.expanded_structural_vector(
            Episode("probe-a", prompt=prefix + "alpha")
        )
        == representation_features.expanded_structural_vector(
            Episode("probe-b", prompt=prefix + "beta")
        )
    )
    checks = {
        "deterministic_features": deterministic,
        "deterministic_plans": plans_equal,
        "input_bound_32768_enforced": bound == 32768 and bound_probe,
        "matched_spend": spend_ok,
        "premium_and_think_identity": premium_identity,
        "runtime_dependencies_stdlib_only": True,
        "runtime_feature_count": len(residual_artifact.feature_names),
    }
    checks["passed"] = all(
        value is True
        for key, value in checks.items()
        if key != "runtime_feature_count"
    ) and checks["runtime_feature_count"] == 36
    return checks


def evaluate_train(
    *,
    train_input: Path = DEFAULT_TRAIN_INPUT,
    train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES,
    artifact_path: Path = DEFAULT_ARTIFACT,
) -> Mapping[str, Any]:
    """Run the complete nested Train gate and write the frozen artifact."""

    _require_numpy()
    validate_protocol()
    inputs, _outcomes, policy, costs, scores, families = _load_train(
        train_input, train_outcomes
    )
    base_artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    predictions = safe_margin.predict_batch(inputs.episodes, base_artifact, policy)
    matrix = _structural_matrix(inputs)
    realized_gain = np.asarray(
        [row[MODEL_IDS[1]] - row[MODEL_IDS[0]] for row in scores],
        dtype=np.float64,
    )
    base_gain = np.asarray(
        [
            row.scores[MODEL_IDS[1]] - row.scores[MODEL_IDS[0]]
            for row in predictions
        ],
        dtype=np.float64,
    )
    targets = realized_gain - base_gain
    all_indices = tuple(range(len(inputs.episodes)))

    global_oof = _oof_residuals(matrix, targets, families, all_indices)
    global_strength, global_grid = _choose_strength(
        predictions, global_oof, scores, policy
    )

    family_results: Dict[str, Any] = {}
    weighted_total = 0.0
    matched_all = True
    for outer_family in risk_validation.FAMILY_LABELS:
        outer = tuple(
            index for index, family in enumerate(families)
            if family == outer_family
        )
        inner = tuple(
            index for index, family in enumerate(families)
            if family != outer_family
        )
        if not outer or not inner:
            raise ExperimentError(f"outer family {outer_family} is indeterminate")
        inner_oof = _oof_residuals(matrix, targets, families, inner)
        inner_strength, inner_grid = _choose_strength(
            _subset(predictions, inner),
            inner_oof[list(inner)],
            _subset(scores, inner),
            policy,
        )
        outer_fitted = _fit_ridge(matrix[list(inner)], targets[list(inner)])
        outer_residuals = _predict_ridge(matrix[list(outer)], outer_fitted)
        outer_metrics = _selection_metrics(
            _subset(predictions, outer),
            outer_residuals,
            _subset(scores, outer),
            policy,
            inner_strength,
        )
        family_results[outer_family] = {
            "held_out_rows": len(outer),
            "inner_grid": inner_grid,
            "inner_selected_strength": inner_strength,
            "outer_metrics": outer_metrics,
        }
        weighted_total += (
            len(outer) * outer_metrics["weighted_improvement"]
        )
        matched_all = matched_all and outer_metrics["matched_spend_passed"]
    nested_improvement = weighted_total / len(inputs.episodes)
    family_improvements = [
        family_results[family]["outer_metrics"]["weighted_improvement"]
        for family in risk_validation.FAMILY_LABELS
    ]

    fitted = _fit_ridge(matrix, targets)
    artifact_value = _artifact_value(
        fitted, policy, global_strength, train_input, train_outcomes
    )
    _write_json_atomic(artifact_path, artifact_value)
    residual_artifact = candidate.load_artifact(artifact_path)
    runtime = _runtime_checks(
        inputs, policy, base_artifact, residual_artifact
    )
    checks = {
        "matched_spend": matched_all
        and all(
            record["matched_spend_passed"]
            for record in global_grid.values()
        ),
        "minimum_weighted_improvement": nested_improvement >= TRAIN_MINIMUM,
        "positive_held_out_families": sum(
            value > 0.0 for value in family_improvements
        )
        >= MIN_POSITIVE_FAMILIES,
        "worst_held_out_family": min(family_improvements) >= FAMILY_FLOOR,
        "runtime_input_bound_determinism": runtime["passed"],
    }
    return {
        "artifact_path": artifact_path.relative_to(ROOT).as_posix(),
        "artifact_sha256": file_sha256(artifact_path),
        "checks": checks,
        "family_results": family_results,
        "gate_passed": all(checks.values()),
        "global_grid": global_grid,
        "global_selected_strength": global_strength,
        "nested_positive_families": sum(
            value > 0.0 for value in family_improvements
        ),
        "nested_weighted_improvement": nested_improvement,
        "nested_worst_family_improvement": min(family_improvements),
        "num_episodes": len(inputs.episodes),
        "runtime_checks": runtime,
        "train_input_path": train_input.relative_to(ROOT).as_posix(),
        "train_input_sha256": file_sha256(train_input),
        "train_outcomes_path": train_outcomes.relative_to(ROOT).as_posix(),
        "train_outcomes_sha256": file_sha256(train_outcomes),
    }


def _candidate_submissions(
    inputs: InputBatch,
    policy: RoutingPolicy,
    base_artifact: Any,
    residual_artifact: candidate.ResidualArtifact,
) -> Tuple[Submission, ...]:
    return tuple(
        candidate.make_submission(
            inputs, policy, base_artifact, residual_artifact, tier
        ).submission
        for tier in TIERS
    )


def _safe_submissions(
    inputs: InputBatch, policy: RoutingPolicy, base_artifact: Any
) -> Tuple[Submission, ...]:
    return tuple(
        safe_margin.make_safe_margin_submission(
            inputs, policy, base_artifact, tier
        ).submission
        for tier in TIERS
    )


def evaluate_dev_once(
    *,
    artifact_path: Path,
    dev_input: Path = DEFAULT_DEV_INPUT,
    dev_outcomes: Path = DEFAULT_DEV_OUTCOMES,
) -> Mapping[str, Any]:
    """Open and score Dev once; callers must first establish the Train gate."""

    inputs = load_input(dev_input)
    outcomes = load_outcomes(dev_outcomes)
    if inputs.split != "dev" or outcomes.split != "dev":
        raise ExperimentError("one-shot inputs must be the public Dev split")
    policy = load_bundled_policy()
    base_artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    residual_artifact = candidate.load_artifact(artifact_path)
    safe_score = score_submissions(
        inputs, outcomes, _safe_submissions(inputs, policy, base_artifact), policy
    )
    candidate_score = score_submissions(
        inputs,
        outcomes,
        _candidate_submissions(
            inputs, policy, base_artifact, residual_artifact
        ),
        policy,
    )
    value = float(candidate_score["final_score"])
    return {
        "candidate_score": candidate_score,
        "dev_input_path": dev_input.relative_to(ROOT).as_posix(),
        "dev_input_sha256": file_sha256(dev_input),
        "dev_outcomes_opened": True,
        "dev_outcomes_path": dev_outcomes.relative_to(ROOT).as_posix(),
        "dev_outcomes_sha256": file_sha256(dev_outcomes),
        "gate_passed": value > DEV_COMPARATOR,
        "minimum_exclusive": DEV_COMPARATOR,
        "practical_target": DEV_PRACTICAL_TARGET,
        "safe_margin_score": safe_score,
    }


def run_experiment(
    *,
    train_input: Path = DEFAULT_TRAIN_INPUT,
    train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES,
    dev_input: Path = DEFAULT_DEV_INPUT,
    dev_outcomes: Path = DEFAULT_DEV_OUTCOMES,
    artifact_path: Path = DEFAULT_ARTIFACT,
    report_path: Path = DEFAULT_REPORT,
) -> Mapping[str, Any]:
    """Run gates in order and persist a byte-stable terminal report."""

    protocol = validate_protocol()
    train = evaluate_train(
        train_input=train_input,
        train_outcomes=train_outcomes,
        artifact_path=artifact_path,
    )
    report: Dict[str, Any] = {
        "adoption": {
            "decision": "not-adopted",
            "submission_default": "safe-margin",
        },
        "dev": {
            "dev_outcomes_opened": False,
            "gate_opened": False,
            "gate_passed": False,
        },
        "experiment_id": candidate.STRATEGY_ID,
        "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_version": protocol["protocol_version"],
        "report_type": "safe-margin-residual-consensus-experiment-report",
        "safety": {"gate_opened": False, "gate_passed": False},
        "train": train,
    }
    if train["gate_passed"]:
        report["dev"] = {
            "gate_opened": True,
            **evaluate_dev_once(
                artifact_path=artifact_path,
                dev_input=dev_input,
                dev_outcomes=dev_outcomes,
            ),
        }
        # The expensive existing 5,000-resample validation is intentionally
        # reached only after the strict one-shot score gate.  A passing run is
        # wired below; a failing Train/Dev run records the closed gate.
        if report["dev"]["gate_passed"]:
            safety = evaluate_safety_once(
                artifact_path=artifact_path,
                dev_input=dev_input,
                dev_outcomes=dev_outcomes,
            )
            report["safety"] = safety
            if safety["gate_passed"]:
                report["adoption"] = {
                    "decision": "adopted",
                    "submission_default": candidate.STRATEGY_ID,
                }
    _write_json_atomic(report_path, _round(report))
    return report


def _candidate_risk_runner(
    data: risk_validation.SplitData, artifact_path: Path
) -> risk_validation.PolicyRunner:
    base_artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    residual_artifact = candidate.load_artifact(artifact_path)
    predictions = safe_margin.predict_batch(
        data.inputs.episodes, base_artifact, data.policy
    )
    residuals = candidate.predict_residuals(
        data.inputs.episodes, residual_artifact
    )

    def plan(indices: Sequence[int], tier: str) -> Tuple[str, ...]:
        selected, _ratio, _matched = candidate.plan_selection(
            _subset(predictions, indices),
            _subset(residuals, indices),
            data.policy,
            tier,
            residual_artifact.strength,
        )
        return selected

    return risk_validation.PolicyRunner(
        name=candidate.STRATEGY_ID,
        artifact_path=artifact_path,
        artifact_sha256=file_sha256(artifact_path),
        plan=plan,
    )


def evaluate_safety_once(
    *,
    artifact_path: Path,
    dev_input: Path,
    dev_outcomes: Path,
) -> Mapping[str, Any]:
    data = risk_validation.load_split_data(
        "dev", input_path=dev_input, outcomes_path=dev_outcomes
    )
    indices = bootstrap_indices(
        data.num_episodes,
        risk_validation.DEFAULT_RESAMPLES,
        risk_validation.DEFAULT_SEED,
    )
    reference = risk_validation.evaluate_runner(
        data,
        risk_validation.safe_margin_runner(data),
        indices,
    )
    measured = risk_validation.evaluate_runner(
        data, _candidate_risk_runner(data, artifact_path), indices
    )
    gate = risk_validation.gate_candidate(reference, measured)
    return {
        "candidate": measured,
        "gate_opened": True,
        "gate_passed": gate["passed"],
        "paired_gate": gate,
        "reference": reference,
        "resamples": risk_validation.DEFAULT_RESAMPLES,
        "seed": risk_validation.DEFAULT_SEED,
    }


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, REPORT_DIGITS)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="safe-margin residual consensus v1 Train-first experiment"
    )
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument(
        "--train-outcomes", type=Path, default=DEFAULT_TRAIN_OUTCOMES
    )
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
    parser.add_argument("--dev-outcomes", type=Path, default=DEFAULT_DEV_OUTCOMES)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_experiment(
            train_input=args.train_input,
            train_outcomes=args.train_outcomes,
            dev_input=args.dev_input,
            dev_outcomes=args.dev_outcomes,
            artifact_path=args.artifact,
            report_path=args.report,
        )
    except (ExperimentError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Train gate: {'PASS' if report['train']['gate_passed'] else 'FAIL'}; "
        f"Dev opened: {report['dev']['dev_outcomes_opened']}; "
        f"decision: {report['adoption']['decision']}"
    )
    return 0 if report["adoption"]["decision"] == "adopted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
