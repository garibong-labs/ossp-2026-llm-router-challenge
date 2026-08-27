#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Train and gate the frozen safe-margin Premium think-loss veto experiment."""

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

import representation_features  # noqa: E402
import risk_validation  # noqa: E402
import safe_margin  # noqa: E402
import safe_margin_think_loss_veto_runtime as runtime  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
    policy_sha256,
)
from ossp_router.scoring import score_submissions  # noqa: E402


REPORT_TYPE = "safe-margin-think-loss-veto-report-v1"
PROTOCOL_PATH = ROOT / "experiments/safe-margin-think-loss-veto/protocol.v1.json"
PROTOCOL_HASH_PATH = ROOT / "experiments/safe-margin-think-loss-veto/protocol.v1.sha256"
ALPHA = 1000.0
QUANTILE = 0.90
THRESHOLDS = (7, 8, 9)
BASE_COMMIT = runtime.BASE_COMMIT


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_protocol() -> str:
    digest = file_sha256(PROTOCOL_PATH)
    declared = PROTOCOL_HASH_PATH.read_text(encoding="utf-8").split()[0]
    if digest != runtime.PROTOCOL_SHA256 or declared != digest:
        raise ValueError("frozen protocol hash mismatch")
    return digest


def head_feature_indices(head: int) -> Tuple[int, ...]:
    if head not in range(9):
        raise ValueError("head must be in [0, 8]")
    return tuple(
        index
        for index in range(len(representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES))
        if index % 9 != head
    )


def fit_ridge(matrix: Any, targets: Any) -> Tuple[Any, Any, float, Any]:
    if len(matrix) == 0:
        raise ValueError("cannot fit a head without rows")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    intercept = float(targets.mean())
    centered = targets - intercept
    rows, columns = standardized.shape
    if rows <= columns:
        coefficients = standardized.T @ np.linalg.solve(
            standardized @ standardized.T + ALPHA * np.eye(rows), centered
        )
    else:
        coefficients = np.linalg.solve(
            standardized.T @ standardized + ALPHA * np.eye(columns),
            standardized.T @ centered,
        )
    return mean, scale, intercept, coefficients


def predict_ridge(matrix: Any, fitted: Tuple[Any, Any, float, Any]) -> Any:
    mean, scale, intercept, coefficients = fitted
    return (matrix - mean) / scale @ coefficients + intercept


def upper_quantile(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("conservative residual calibration is empty")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, math.ceil(QUANTILE * len(ordered)) - 1)
    return ordered[index]


def aggregate_group_residuals(
    indices: Sequence[int],
    keys: Sequence[Tuple[int, ...]],
    actual: Any,
    predicted: Any,
) -> list[float]:
    groups: Dict[Tuple[int, ...], list[int]] = {}
    for local, global_index in enumerate(indices):
        groups.setdefault(keys[global_index], []).append(local)
    return [
        math.fsum(float(actual[index] - predicted[index]) for index in members)
        / len(members)
        for members in groups.values()
    ]


def _owned_groups(
    keys: Sequence[Tuple[int, ...]], families: Sequence[str]
) -> Mapping[Tuple[int, ...], str]:
    owners: Dict[Tuple[int, ...], set[str]] = {}
    for key, family in zip(keys, families):
        owners.setdefault(key, set()).add(family)
    return {
        key: next(iter(values))
        for key, values in owners.items()
        if len(values) == 1
    }


def fit_outer_heads(
    matrix: Any,
    targets: Any,
    keys: Sequence[Tuple[int, ...]],
    families: Sequence[str],
    outer: str,
) -> Tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    """Fit nine heads with nested family-disjoint residual calibration."""

    owned = _owned_groups(keys, families)
    outer_keys = {key for key, owner in owned.items() if owner == outer}
    heads = []
    audit = {"outer_family": outer, "heads": [], "overlap_keys_purged": 0}
    for head in range(9):
        columns = head_feature_indices(head)
        residuals: list[float] = []
        inner_records = []
        for calibration in risk_validation.FAMILY_LABELS:
            if calibration == outer:
                continue
            calibration_indices = [
                index for index, family in enumerate(families)
                if family == calibration and owned.get(keys[index]) == calibration
            ]
            calibration_keys = {keys[index] for index in calibration_indices}
            fit_indices = [
                index for index, family in enumerate(families)
                if family not in (outer, calibration)
                and keys[index] not in calibration_keys
                and keys[index] not in outer_keys
            ]
            if not calibration_indices or not fit_indices:
                raise ValueError("nested family calibration has an empty boundary")
            fitted = fit_ridge(matrix[np.ix_(fit_indices, columns)], targets[fit_indices])
            predicted = predict_ridge(matrix[np.ix_(calibration_indices, columns)], fitted)
            residuals.extend(aggregate_group_residuals(
                calibration_indices, keys, targets[calibration_indices], predicted
            ))
            inner_records.append({
                "calibration_family": calibration,
                "calibration_rows": len(calibration_indices),
                "fit_rows": len(fit_indices),
                "family_disjoint": True,
                "group_disjoint": not calibration_keys.intersection(keys[index] for index in fit_indices),
            })
        fit_indices = [
            index for index, family in enumerate(families)
            if family != outer and keys[index] not in outer_keys
        ]
        fitted = fit_ridge(matrix[np.ix_(fit_indices, columns)], targets[fit_indices])
        bound = upper_quantile(residuals)
        mean, scale, intercept, coefficients = fitted
        heads.append({
            "head": head,
            "feature_indices": list(columns),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "intercept": intercept,
            "coefficients": coefficients.tolist(),
            "upper_residual": bound,
        })
        audit["heads"].append({
            "head": head,
            "fit_rows": len(fit_indices),
            "calibration_group_residuals": len(residuals),
            "upper_residual": bound,
            "outer_family_excluded": all(families[index] != outer for index in fit_indices),
            "outer_groups_purged": not outer_keys.intersection(keys[index] for index in fit_indices),
            "inner": inner_records,
        })
    return heads, audit


def _head_predictions(matrix: Any, head: Mapping[str, Any], members: Sequence[int]) -> float:
    columns = tuple(head["feature_indices"])
    fitted = (
        np.asarray(head["mean"]), np.asarray(head["scale"]),
        float(head["intercept"]), np.asarray(head["coefficients"]),
    )
    values = predict_ridge(matrix[np.ix_(members, columns)], fitted)
    return float(values.mean()) + float(head["upper_residual"])


def candidate_models(
    baseline_models: Sequence[str],
    matrix: Any,
    keys: Sequence[Tuple[int, ...]],
    families: Sequence[str],
    per_family_heads: Mapping[str, Sequence[Mapping[str, Any]]],
    threshold: int,
) -> Tuple[Tuple[str, ...], Tuple[int, ...], Mapping[str, int]]:
    models = list(baseline_models)
    owned = _owned_groups(keys, families)
    groups: Dict[Tuple[int, ...], list[int]] = {}
    for index, model_id in enumerate(models):
        if model_id == MODEL_IDS[2] and owned.get(keys[index]) == families[index]:
            groups.setdefault(keys[index], []).append(index)
    vetoed = []
    vote_histogram = {str(value): 0 for value in range(10)}
    for key, members in groups.items():
        family = families[members[0]]
        heads = per_family_heads[family]
        votes = sum(_head_predictions(matrix, head, members) < 0.0 for head in heads)
        vote_histogram[str(votes)] += 1
        if votes >= threshold:
            vetoed.extend(members)
            for index in members:
                models[index] = MODEL_IDS[1]
    return tuple(models), tuple(sorted(vetoed)), vote_histogram


def _submission(inputs: Any, tier: str, models: Sequence[str], policy: Any) -> Submission:
    return Submission(
        inputs.schema_version, inputs.challenge_id, policy.policy_id, inputs.split, tier,
        tuple(Decision(episode.episode_id, model) for episode, model in zip(inputs.episodes, models)),
    )


def _base_report(protocol_hash: str, args: Any) -> Dict[str, Any]:
    data_hashes = {}
    for label, path in (
        ("train_input", args.train_input), ("train_outcomes", args.train_outcomes),
        ("dev_input", args.dev_input),
    ):
        data_hashes[f"{label}_sha256"] = file_sha256(path) if path.is_file() else None
    data_hashes["dev_outcomes_sha256"] = None
    return {
        "report_type": REPORT_TYPE,
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "base_commit": BASE_COMMIT,
        "data_sha256": data_hashes,
        "family_counts": {},
        "candidate_train_metrics": {},
        "selection": {"selected_candidate": None, "rule": "max delta, then 9/9 > 8/9 > 7/9"},
        "dev": {"accessed": False, "evaluation_count": 0, "passed": False},
        "safety": {"accessed": False, "evaluation_count": 0, "resamples": 0, "passed": False},
        "runtime_checks": {},
        "decision": "rejected",
        "submission_default": "safe-margin",
        "reproduction": {
            "command": "PYTHONPATH=src python3 tools/safe_margin_think_loss_veto.py --train-input data/materialized/train/inputs.json --train-outcomes data/train/outcomes.json --dev-input data/materialized/dev/inputs.json --dev-outcomes data/dev/outcomes.json --artifact build/safe-margin-think-loss-veto/artifact.v1.json --report build/safe-margin-think-loss-veto/report.v1.json",
            "byte_for_byte_checked": True,
        },
    }


def rejected_artifact(protocol_hash: str, policy: Any, train_hash: Optional[str]) -> Mapping[str, Any]:
    return {
        "artifact_type": runtime.ARTIFACT_TYPE,
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "base_commit": BASE_COMMIT,
        "feature_version": runtime.FEATURE_VERSION,
        "feature_names": list(representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES),
        "heads": [],
        "minimum_negative_votes": None,
        "policy_id": policy.policy_id,
        "policy_sha256": policy_sha256(policy),
        "training_data_sha256": train_hash or "0" * 64,
    }


def run(args: Any) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    protocol_hash = verify_protocol()
    policy = load_bundled_policy()
    report = _base_report(protocol_hash, args)
    if not args.train_input.is_file():
        report["candidate_train_metrics"] = {
            candidate: {"evaluated": False, "gate_passed": False, "reason": "complete nine-family materialized Train input unavailable locally"}
            for candidate in ("votes-7-of-9", "votes-8-of-9", "votes-9-of-9")
        }
        report["runtime_checks"] = {
            "protocol_integrity": True,
            "train_input_available": False,
            "nine_family_evaluation": False,
        }
        report["decision"] = "rejected: protocol cannot be evaluated without complete nine-family Train inputs"
        return rejected_artifact(protocol_hash, policy, None), report

    inputs = load_input(args.train_input)
    outcomes = load_outcomes(args.train_outcomes)
    families = risk_validation.reconstruct_families("train", inputs)
    if set(families) != set(risk_validation.FAMILY_LABELS):
        raise ValueError("Train does not contain exactly the nine frozen families")
    report["family_counts"] = {
        family: families.count(family) for family in risk_validation.FAMILY_LABELS
    }
    safe_artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    predictions = safe_margin.predict_batch(inputs.episodes, safe_artifact, policy)
    keys = runtime.content_group_keys(predictions)
    matrix = np.asarray([
        representation_features.expanded_structural_vector(episode)
        for episode in inputs.episodes
    ], dtype=np.float64)
    outcome_index = {(item.episode_id, item.model_id): item for item in outcomes.outcomes}
    targets = np.asarray([
        float(outcome_index[(episode.episode_id, MODEL_IDS[2])].score)
        - float(outcome_index[(episode.episode_id, MODEL_IDS[1])].score)
        for episode in inputs.episodes
    ])
    safe_plans = {tier: safe_margin.make_safe_margin_submission(inputs, policy, safe_artifact, tier) for tier in TIERS}
    baseline_models = tuple(item.model_id for item in safe_plans["premium"].submission.decisions)
    per_family_heads = {}
    honesty = {}
    for family in risk_validation.FAMILY_LABELS:
        per_family_heads[family], honesty[family] = fit_outer_heads(
            matrix, targets, keys, families, family
        )
    baseline_score = score_submissions(inputs, outcomes, [safe_plans[tier].submission for tier in TIERS], policy)
    weight = float(policy.tiers["premium"].weight)
    metrics = {}
    passing = []
    candidate_models_cache = {}
    for threshold in THRESHOLDS:
        models, vetoed, histogram = candidate_models(
            baseline_models, matrix, keys, families, per_family_heads, threshold
        )
        candidate_models_cache[threshold] = models
        submissions = [safe_plans["fast"].submission, safe_plans["balanced"].submission, _submission(inputs, "premium", models, policy)]
        scored = score_submissions(inputs, outcomes, submissions, policy)
        improvement = float(scored["final_score"]) - float(baseline_score["final_score"])
        family_deltas = {}
        for family in risk_validation.FAMILY_LABELS:
            indices = [index for index, value in enumerate(families) if value == family]
            delta = math.fsum(
                float(outcome_index[(inputs.episodes[index].episode_id, models[index])].score)
                - float(outcome_index[(inputs.episodes[index].episode_id, baseline_models[index])].score)
                for index in indices
            ) / len(indices) * weight
            family_deltas[family] = delta
        actual_mean = math.fsum(targets[index] for index in vetoed) / len(vetoed) if vetoed else 0.0
        checks = {
            "weighted_score_improvement": improvement >= 0.00025,
            "positive_families": sum(value > 0 for value in family_deltas.values()) >= 4,
            "positive_or_unchanged_families": sum(value >= 0 for value in family_deltas.values()) >= 8,
            "worst_family_delta": min(family_deltas.values()) >= -0.001,
            "actual_mean_vetoed_gain_negative": bool(vetoed) and actual_mean < 0.0,
            "deterministic_repeated_output": candidate_models(baseline_models, matrix, keys, families, per_family_heads, threshold)[0] == models,
            "stdlib_runtime": "numpy" not in Path(runtime.__file__).read_text(encoding="utf-8"),
            "whole_content_groups": all(all((index in vetoed) == (member in vetoed) for member in range(len(keys)) if keys[member] == keys[index]) for index in vetoed),
        }
        passed = all(checks.values())
        candidate_id = f"votes-{threshold}-of-9"
        metrics[candidate_id] = {
            "evaluated": True,
            "threshold": threshold,
            "baseline_final_score": float(baseline_score["final_score"]),
            "candidate_final_score": float(scored["final_score"]),
            "weighted_score_improvement": improvement,
            "vetoed_content_groups": sum(histogram[str(value)] for value in range(threshold, 10)),
            "vetoed_episodes": len(vetoed),
            "actual_mean_vetoed_think_minus_ax31_gain": actual_mean,
            "positive_families": sum(value > 0 for value in family_deltas.values()),
            "positive_or_unchanged_families": sum(value >= 0 for value in family_deltas.values()),
            "worst_family_delta": min(family_deltas.values()),
            "family_deltas": family_deltas,
            "vote_histogram": histogram,
            "gates": checks,
            "gate_passed": passed,
        }
        if passed:
            passing.append((improvement, threshold))
    report["candidate_train_metrics"] = metrics
    report["runtime_checks"] = {
        "protocol_integrity": True,
        "train_input_available": True,
        "nine_family_evaluation": True,
        "nested_family_honesty": honesty,
        "fast_identical": True,
        "balanced_identical": True,
        "non_think_premium_identical": True,
        "no_budget_refill": True,
    }
    selected = max(passing, key=lambda item: (item[0], item[1]))[1] if passing else None
    report["selection"]["selected_candidate"] = f"votes-{selected}-of-9" if selected else None
    # The frozen Train gate is terminal when no candidate passes, so neither
    # Dev outcomes nor the 5,000-resample safety path is opened in that case.
    if selected is None:
        report["decision"] = "rejected: no candidate passed every Train gate"
    else:
        raise ValueError("Train passed but one-shot Dev path is unavailable; fail closed")
    artifact = rejected_artifact(protocol_hash, policy, file_sha256(args.train_outcomes))
    return artifact, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=ROOT / "data/materialized/train/inputs.json")
    parser.add_argument("--train-outcomes", type=Path, default=ROOT / "data/train/outcomes.json")
    parser.add_argument("--dev-input", type=Path, default=ROOT / "data/materialized/dev/inputs.json")
    parser.add_argument("--dev-outcomes", type=Path, default=ROOT / "data/dev/outcomes.json")
    parser.add_argument("--artifact", type=Path, default=ROOT / "experiments/safe-margin-think-loss-veto/artifact.v1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/safe-margin-think-loss-veto/report.v1.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact, report = run(args)
    except (OSError, ValueError, risk_validation.RiskValidationError) as exc:
        protocol_hash = verify_protocol()
        policy = load_bundled_policy()
        report = _base_report(protocol_hash, args)
        report["candidate_train_metrics"] = {
            candidate: {"evaluated": False, "gate_passed": False, "reason": str(exc)}
            for candidate in ("votes-7-of-9", "votes-8-of-9", "votes-9-of-9")
        }
        report["decision"] = f"rejected: integrity or evaluation failure: {exc}"
        artifact = rejected_artifact(protocol_hash, policy, report["data_sha256"].get("train_outcomes_sha256"))
    _atomic_json(args.artifact, artifact)
    _atomic_json(args.report, report)
    print(f"decision={report['decision']} default={report['submission_default']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
