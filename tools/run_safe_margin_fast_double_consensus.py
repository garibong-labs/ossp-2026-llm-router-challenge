#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Run the one frozen safe-margin Fast double-consensus experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "baselines", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import representation_features  # noqa: E402
import risk_validation  # noqa: E402
import safe_margin  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    Message,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from ossp_router.scoring import score_submissions  # noqa: E402
from stress_safe_margin import bootstrap_indices, outcome_tables  # noqa: E402

EXPERIMENT_ID = "safe-margin-fast-double-consensus-v1"
REPORT_TYPE = "safe-margin-fast-double-consensus-report-v1"
BASE_COMMIT = "3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9"
EXPERIMENT_DIR = ROOT / "experiments/safe-margin-fast-double-consensus"
PROTOCOL_PATH = EXPERIMENT_DIR / "protocol.v1.json"
ARTIFACT_PATH = EXPERIMENT_DIR / "residual-consensus.v1.json"
DEFAULT_REPORT = EXPERIMENT_DIR / "report.v1.json"
EXPECTED_PROTOCOL_SHA256 = "f7552ed172d2907766f3fb5fb4c3dd3bea24e1f9826d1bcd0b0bf69d9dca930d"
EXPECTED_ARTIFACT_SHA256 = "cf33622f9c6d31b21a44fe65530c562b4d8519e797029b8ddd69508ea2e3b964"
DEFAULT_TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"
DEFAULT_TRAIN_OUTCOMES = ROOT / "data/train/outcomes.json"
DEFAULT_DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"
DEFAULT_DEV_OUTCOMES = ROOT / "data/dev/outcomes.json"
BUCKETS_PER_OCTAVE = 3
RESIDUAL_CLIP = (-0.1, 0.1)
DEV_COMPARATOR = Decimal("0.673182")
REPORT_DIGITS = 12
LIGHT, AX31, THINK = MODEL_IDS


class ExperimentError(RuntimeError):
    """Frozen evidence is missing, malformed, inconsistent, or uncertain."""


@dataclass(frozen=True)
class ResidualArtifact:
    feature_names: Tuple[str, ...]
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    intercept: float
    coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class EvaluationData:
    inputs: InputBatch
    outcomes: OutcomeBatch
    policy: RoutingPolicy
    costs: Tuple[Mapping[str, float], ...]
    scores: Tuple[Mapping[str, float], ...]
    families: Tuple[str, ...]
    input_path: Path
    outcomes_path: Path


@dataclass(frozen=True)
class CandidatePlan:
    selected: Tuple[str, ...]
    predicted_ratio: float
    baseline_ratio: float
    promoted: int
    swaps: int
    predicted_increment: float
    baseline_increment: float
    fallback_used: bool = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    content = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_protocol() -> Mapping[str, Any]:
    if file_sha256(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise ExperimentError("frozen protocol SHA-256 mismatch")
    if file_sha256(ARTIFACT_PATH) != EXPECTED_ARTIFACT_SHA256:
        raise ExperimentError("frozen residual artifact SHA-256 mismatch")
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    try:
        valid = (
            value["base"]["commit"] == BASE_COMMIT
            and value["artifact"]["sha256"] == EXPECTED_ARTIFACT_SHA256
            and value["candidate"]["efficiency_buckets_per_octave"] == 3
            and value["candidate"]["scope"] == "fast light->ax31 only"
            and value["candidate"]["tuning_sweep"] is False
            and value["default_runtime"]["changed"] is False
            and value["default_runtime"]["efficiency_buckets_per_octave"]
            == safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE == 1
            and tuple(value["train_gate"]["benchmark_families"])
            == risk_validation.FAMILY_LABELS
            and value["safety_gate"]["resamples"]
            == risk_validation.DEFAULT_RESAMPLES == 5000
            and value["safety_gate"]["seed"] == risk_validation.DEFAULT_SEED
            and value["safety_gate"]["tolerance"] == 0.0
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ExperimentError("frozen protocol contents mismatch")
    return value


def _finite_tuple(value: Any, name: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ExperimentError(f"{name} length mismatch")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ExperimentError(f"{name} must contain finite numbers")
    return result


def load_residual_artifact() -> ResidualArtifact:
    if file_sha256(ARTIFACT_PATH) != EXPECTED_ARTIFACT_SHA256:
        raise ExperimentError("frozen residual artifact SHA-256 mismatch")
    value = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    names = tuple(representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES)
    if (
        value.get("strategy_id") != "safe-margin-residual-consensus-v1"
        or value.get("representation") != "B-expanded-structural"
        or tuple(value.get("feature_names", ())) != names
        or value.get("policy_sha256")
        != "7c892c423da5fa762e7e1a93b9fa071be51e259b65d2b63a5ba434c4342d7a8e"
    ):
        raise ExperimentError("frozen residual artifact contract mismatch")
    scale = _finite_tuple(value.get("feature_scale"), "feature_scale", len(names))
    if any(item <= 0 for item in scale):
        raise ExperimentError("feature_scale must be positive")
    intercept = float(value.get("intercept", float("nan")))
    if not math.isfinite(intercept):
        raise ExperimentError("intercept must be finite")
    return ResidualArtifact(
        names,
        _finite_tuple(value.get("feature_mean"), "feature_mean", len(names)),
        scale,
        intercept,
        _finite_tuple(value.get("coefficients"), "coefficients", len(names)),
    )


def _content_only_episode(episode: Episode) -> Episode:
    if episode.prompt is not None:
        return Episode("", prompt=episode.prompt)
    assert episode.messages is not None
    return Episode(
        "", messages=tuple(Message(item.role, item.content) for item in episode.messages)
    )


def predict_residual(episode: Episode, artifact: ResidualArtifact) -> float:
    vector = representation_features.expanded_structural_vector(
        _content_only_episode(episode)
    )
    if len(vector) != len(artifact.feature_names) or not all(
        math.isfinite(item) for item in vector
    ):
        raise ExperimentError("missing or malformed residual features")
    value = artifact.intercept + math.fsum(
        coefficient * ((feature - mean) / scale)
        for feature, mean, scale, coefficient in zip(
            vector,
            artifact.feature_mean,
            artifact.feature_scale,
            artifact.coefficients,
        )
    )
    if not math.isfinite(value):
        raise ExperimentError("non-finite residual prediction")
    return min(RESIDUAL_CLIP[1], max(RESIDUAL_CLIP[0], value))


def predict_residuals(
    episodes: Sequence[Episode], artifact: ResidualArtifact
) -> Tuple[float, ...]:
    return tuple(predict_residual(episode, artifact) for episode in episodes)


def _bucket(efficiency: float) -> int:
    if not math.isfinite(efficiency) or efficiency <= safe_margin.MIN_EFFICIENCY:
        raise ExperimentError("missing or malformed x3 efficiency signal")
    return int(math.floor(math.log2(efficiency) * BUCKETS_PER_OCTAVE))


def _eligible_groups(
    predictions: Sequence[safe_margin.EpisodePrediction],
) -> Mapping[Tuple[int, ...], Tuple[int, ...]]:
    config = safe_margin.TIER_PLAN_CONFIGS["fast"]
    light_total = math.fsum(item.costs[LIGHT] for item in predictions)
    if not predictions or not math.isfinite(light_total) or light_total <= 0:
        raise ExperimentError("invalid prediction evidence")
    mean_light = light_total / len(predictions)
    groups: Dict[Tuple[int, ...], list[int]] = {}
    for index, prediction in enumerate(predictions):
        gain = prediction.scores[AX31] - prediction.scores[LIGHT]
        increment = prediction.costs[AX31] - prediction.costs[LIGHT]
        step_ratio = prediction.costs[AX31] / prediction.costs[LIGHT]
        step_load = increment / mean_light
        if not all(math.isfinite(value) for value in (gain, increment, step_ratio, step_load)):
            raise ExperimentError("non-finite x3 prediction evidence")
        if (
            gain < config.ax31_min_gain
            or increment <= 0
            or step_ratio > config.ax31_max_step_ratio
            or step_load > config.ax31_max_step_load
        ):
            continue
        key = (_bucket(gain / step_load),) + prediction.signature
        groups.setdefault(key, []).append(index)
    return {key: tuple(indices) for key, indices in groups.items()}


def matched_fast_plan(
    predictions: Sequence[safe_margin.EpisodePrediction],
    residuals: Sequence[float],
    policy: RoutingPolicy,
) -> CandidatePlan:
    """Apply disjoint Pareto-consensus swaps to the exact x1 Fast plan."""

    baseline, baseline_ratio, _stages = safe_margin.plan_selection(
        predictions, policy, "fast"
    )

    def fallback() -> CandidatePlan:
        increment = math.fsum(
            item.costs[AX31] - item.costs[LIGHT]
            for item, model in zip(predictions, baseline)
            if model == AX31
        )
        return CandidatePlan(
            baseline,
            baseline_ratio,
            baseline_ratio,
            baseline.count(AX31),
            0,
            increment,
            increment,
            True,
        )

    if len(residuals) != len(predictions) or not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in residuals
    ):
        return fallback()
    try:
        groups = _eligible_groups(predictions)
    except (ExperimentError, OverflowError, ValueError, ZeroDivisionError):
        return fallback()
    if any(
        any(baseline[i] == AX31 for i in members)
        and not all(baseline[i] == AX31 for i in members)
        for members in groups.values()
    ):
        return fallback()
    baseline_selected = {
        key for key, members in groups.items() if all(baseline[i] == AX31 for i in members)
    }

    def load(key: Tuple[int, ...]) -> float:
        return math.fsum(
            predictions[i].costs[AX31] - predictions[i].costs[LIGHT]
            for i in groups[key]
        )

    def residual_rank(key: Tuple[int, ...]) -> float:
        values = tuple(float(residuals[i]) for i in groups[key])
        return math.fsum(values) / len(values)

    ranks = {key: residual_rank(key) for key in groups}
    if not all(math.isfinite(value) for value in ranks.values()):
        return fallback()
    pairs = []
    for incoming in groups.keys() - baseline_selected:
        for outgoing in baseline_selected:
            if (
                len(groups[incoming]) == len(groups[outgoing])
                and incoming[0] > outgoing[0]
                and ranks[incoming] > ranks[outgoing]
                and load(incoming) <= load(outgoing)
            ):
                pairs.append(
                    (
                        -incoming[0],
                        -ranks[incoming],
                        outgoing[0],
                        ranks[outgoing],
                        incoming[1:],
                        outgoing[1:],
                        incoming,
                        outgoing,
                    )
                )
    used = set()
    selected = list(baseline)
    swaps = 0
    for *_order, incoming, outgoing in sorted(pairs):
        if incoming in used or outgoing in used:
            continue
        for index in groups[outgoing]:
            selected[index] = LIGHT
        for index in groups[incoming]:
            selected[index] = AX31
        used.update((incoming, outgoing))
        swaps += 1

    light_total = math.fsum(item.costs[LIGHT] for item in predictions)
    baseline_increment = math.fsum(
        item.costs[AX31] - item.costs[LIGHT]
        for item, model in zip(predictions, baseline)
        if model == AX31
    )
    candidate_increment = math.fsum(
        item.costs[AX31] - item.costs[LIGHT]
        for item, model in zip(predictions, selected)
        if model == AX31
    )
    if (
        selected.count(AX31) != baseline.count(AX31)
        or selected.count(THINK) != 0
        or candidate_increment > baseline_increment
    ):
        return fallback()
    return CandidatePlan(
        tuple(selected),
        (light_total + candidate_increment) / light_total,
        baseline_ratio,
        selected.count(AX31),
        swaps,
        candidate_increment,
        baseline_increment,
        False,
    )


def candidate_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: Any,
    tier: str,
    predictions: Optional[Sequence[safe_margin.EpisodePrediction]] = None,
    residuals: Optional[Sequence[float]] = None,
) -> Tuple[Submission, Optional[CandidatePlan]]:
    baseline = safe_margin.make_safe_margin_submission(inputs, policy, artifact, tier)
    if tier != "fast":
        return baseline.submission, None
    prediction_values = tuple(predictions) if predictions is not None else tuple(
        safe_margin.predict_batch(inputs.episodes, artifact, policy)
    )
    residual_values = tuple(residuals) if residuals is not None else tuple(
        predict_residuals(inputs.episodes, load_residual_artifact())
    )
    plan = matched_fast_plan(prediction_values, residual_values, policy)
    return (
        Submission(
            schema_version=inputs.schema_version,
            challenge_id=inputs.challenge_id,
            policy_id=policy.policy_id,
            split=inputs.split,
            tier=tier,
            decisions=tuple(
                Decision(episode.episode_id, model)
                for episode, model in zip(inputs.episodes, plan.selected)
            ),
        ),
        plan,
    )


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise InvalidOperation
    result = Decimal(str(value))
    if not result.is_finite():
        raise InvalidOperation
    return result


def decide_train_gate(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    names = risk_validation.FAMILY_LABELS
    checks = {name: False for name in (
        "weighted_score_delta_positive", "fast_quality_delta_positive",
        "positive_families", "nonnegative_families", "worst_family_delta",
        "balanced_premium_exact_x1", "fast_model_counts_exact_x1",
        "candidate_fast_predicted_spend_lte_x1", "deterministic_repeated_output",
    )}
    try:
        deltas = evidence["family_deltas"]
        if set(deltas) != set(names):
            raise InvalidOperation
        values = [_decimal(deltas[name]) for name in names]
        positive = sum(value > 0 for value in values)
        nonnegative = sum(value >= 0 for value in values)
        worst = min(values)
        checks = {
            "weighted_score_delta_positive": _decimal(evidence["weighted_score_delta"]) > 0,
            "fast_quality_delta_positive": _decimal(evidence["fast_quality_delta"]) > 0,
            "positive_families": positive >= 4,
            "nonnegative_families": nonnegative == 9,
            "worst_family_delta": worst >= 0,
            "balanced_premium_exact_x1": evidence["balanced_premium_exact_x1"] is True,
            "fast_model_counts_exact_x1": evidence["fast_model_counts_exact_x1"] is True,
            "candidate_fast_predicted_spend_lte_x1": evidence["candidate_fast_predicted_spend_lte_x1"] is True,
            "deterministic_repeated_output": evidence["deterministic_repeated_output"] is True,
        }
        return {
            "checks": checks, "gate_passed": all(checks.values()),
            "malformed": False, "positive_families": positive,
            "nonnegative_families": nonnegative, "worst_family_delta": str(worst),
        }
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return {"checks": checks, "gate_passed": False, "malformed": True}


def decide_dev_gate(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = {name: False for name in (
        "score_strictly_above_threshold", "balanced_premium_exact_x1",
        "fast_model_counts_exact_x1", "candidate_fast_predicted_spend_lte_x1",
    )}
    try:
        checks = {
            "score_strictly_above_threshold": _decimal(evidence["candidate_score"]) > DEV_COMPARATOR,
            "balanced_premium_exact_x1": evidence["balanced_premium_exact_x1"] is True,
            "fast_model_counts_exact_x1": evidence["fast_model_counts_exact_x1"] is True,
            "candidate_fast_predicted_spend_lte_x1": evidence["candidate_fast_predicted_spend_lte_x1"] is True,
        }
        return {"checks": checks, "gate_passed": all(checks.values()), "malformed": False}
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return {"checks": checks, "gate_passed": False, "malformed": True}


def decide_safety_gate(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Any]:
    checks: Dict[str, bool] = {}
    malformed = False
    try:
        for tier in TIERS:
            ref = reference["tiers"][tier]
            measured = candidate["tiers"][tier]
            checks[f"{tier}.bootstrap_cap_breaches"] = (
                int(measured["bootstrap"]["cap_breaches"]) == 0
            )
            checks[f"{tier}.whole_split_budget"] = (
                measured["whole_split"]["budget_passed"] is True
            )
            for family in risk_validation.FAMILY_LABELS:
                holdout = measured["family_holdouts"][family]
                checks[f"{tier}.holdout.{family}.budget"] = (
                    holdout["evaluated"] is True and holdout["budget_passed"] is True
                )
            for statistic in ("p99", "max"):
                checks[f"{tier}.{statistic}_no_worse"] = _decimal(
                    measured["bootstrap"]["cost_ratio"][statistic]
                ) <= _decimal(ref["bootstrap"]["cost_ratio"][statistic])
    except (InvalidOperation, KeyError, TypeError, ValueError, OverflowError):
        malformed = True
    return {
        "checks": checks,
        "gate_passed": bool(checks) and not malformed and all(checks.values()),
        "malformed": malformed,
    }


def _load_split_once(split: str, input_path: Path, outcomes_path: Path) -> EvaluationData:
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    if inputs.split != split or outcomes.split != split:
        raise ExperimentError(f"expected public {split}")
    policy = load_bundled_policy()
    costs, scores = outcome_tables(inputs, outcomes, policy)
    families = risk_validation.reconstruct_families(split, inputs)
    if set(families) != set(risk_validation.FAMILY_LABELS):
        raise ExperimentError(f"{split} does not contain all frozen families")
    return EvaluationData(
        inputs, outcomes, policy, costs, scores, families, input_path, outcomes_path
    )


def _plans(
    data: EvaluationData,
) -> Tuple[Tuple[Submission, ...], Tuple[Submission, ...], CandidatePlan]:
    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    residual_artifact = load_residual_artifact()
    predictions = tuple(
        safe_margin.predict_batch(data.inputs.episodes, artifact, data.policy)
    )
    residuals = predict_residuals(data.inputs.episodes, residual_artifact)
    baseline = tuple(
        safe_margin.make_safe_margin_submission(
            data.inputs, data.policy, artifact, tier
        ).submission
        for tier in TIERS
    )
    pairs = tuple(
        candidate_submission(
            data.inputs, data.policy, artifact, tier, predictions, residuals
        )
        for tier in TIERS
    )
    if pairs[0][1] is None:
        raise ExperimentError("missing Fast plan")
    return baseline, tuple(pair[0] for pair in pairs), pairs[0][1]


def _models(submission: Submission) -> Tuple[str, ...]:
    return tuple(decision.model_id for decision in submission.decisions)


def _quality(data: EvaluationData, submission: Submission) -> Decimal:
    chosen = {decision.episode_id: decision.model_id for decision in submission.decisions}
    lookup = {(row.episode_id, row.model_id): row.score for row in data.outcomes.outcomes}
    with localcontext() as context:
        context.prec = 80
        return sum(
            (
                lookup[(episode.episode_id, chosen[episode.episode_id])]
                for episode in data.inputs.episodes
            ),
            Decimal(0),
        ) / Decimal(len(data.inputs.episodes))


def _family_deltas(
    data: EvaluationData,
    baseline: Sequence[Submission],
    candidate: Sequence[Submission],
) -> Mapping[str, str]:
    lookup = {(row.episode_id, row.model_id): row.score for row in data.outcomes.outcomes}
    base_models = {
        item.tier: {decision.episode_id: decision.model_id for decision in item.decisions}
        for item in baseline
    }
    candidate_models = {
        item.tier: {decision.episode_id: decision.model_id for decision in item.decisions}
        for item in candidate
    }
    result: Dict[str, str] = {}
    with localcontext() as context:
        context.prec = 80
        for family in risk_validation.FAMILY_LABELS:
            indices = [i for i, label in enumerate(data.families) if label == family]
            if not indices:
                raise ExperimentError(f"family {family} has no rows")
            delta = Decimal(0)
            for tier in TIERS:
                total = Decimal(0)
                for index in indices:
                    episode_id = data.inputs.episodes[index].episode_id
                    total += (
                        lookup[(episode_id, candidate_models[tier][episode_id])]
                        - lookup[(episode_id, base_models[tier][episode_id])]
                    )
                delta += data.policy.tiers[tier].weight * total / Decimal(len(indices))
            result[family] = str(delta)
    return result


def _common_evidence(data: EvaluationData) -> Mapping[str, Any]:
    baseline, candidate, fast_plan = _plans(data)
    baseline_score = score_submissions(data.inputs, data.outcomes, baseline, data.policy)
    candidate_score = score_submissions(data.inputs, data.outcomes, candidate, data.policy)
    repeated = _plans(data)[1] == candidate
    return {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_score": baseline_score,
        "candidate_score_detail": candidate_score,
        "candidate_score": candidate_score["final_score"],
        "weighted_score_delta": str(
            _decimal(candidate_score["final_score"])
            - _decimal(baseline_score["final_score"])
        ),
        "fast_quality_delta": str(
            _quality(data, candidate[0]) - _quality(data, baseline[0])
        ),
        "balanced_premium_exact_x1": all(
            _models(candidate[i]) == _models(baseline[i]) for i in (1, 2)
        ),
        "fast_model_counts_exact_x1": sorted(_models(candidate[0]))
        == sorted(_models(baseline[0])),
        "candidate_fast_predicted_spend_lte_x1": (
            fast_plan.predicted_increment <= fast_plan.baseline_increment
        ),
        "deterministic_repeated_output": repeated,
        "fast_plan": {
            "baseline_predicted_ratio": fast_plan.baseline_ratio,
            "candidate_predicted_ratio": fast_plan.predicted_ratio,
            "baseline_increment": fast_plan.baseline_increment,
            "candidate_increment": fast_plan.predicted_increment,
            "promoted_episodes": fast_plan.promoted,
            "matched_swaps": fast_plan.swaps,
            "fallback_used": fast_plan.fallback_used,
        },
    }


def evaluate_train(
    train_input: Path = DEFAULT_TRAIN_INPUT,
    train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES,
) -> Mapping[str, Any]:
    data = _load_split_once("train", train_input, train_outcomes)
    evidence = _common_evidence(data)
    family_deltas = _family_deltas(data, evidence["baseline"], evidence["candidate"])
    gate = decide_train_gate({**evidence, "family_deltas": family_deltas})
    return {
        "accessed": True,
        "load_count": 1,
        "evaluation_count": 1,
        "baseline_score": evidence["baseline_score"],
        "candidate_score": evidence["candidate_score_detail"],
        "weighted_score_delta": evidence["weighted_score_delta"],
        "fast_quality_delta": evidence["fast_quality_delta"],
        "family_deltas": family_deltas,
        "family_counts": {
            name: data.families.count(name) for name in risk_validation.FAMILY_LABELS
        },
        "balanced_premium_exact_x1": evidence["balanced_premium_exact_x1"],
        "fast_model_counts_exact_x1": evidence["fast_model_counts_exact_x1"],
        "candidate_fast_predicted_spend_lte_x1": evidence["candidate_fast_predicted_spend_lte_x1"],
        "deterministic_repeated_output": evidence["deterministic_repeated_output"],
        "fast_plan": evidence["fast_plan"],
        "input_path": _repo_path(train_input),
        "input_sha256": file_sha256(train_input),
        "outcomes_path": _repo_path(train_outcomes),
        "outcomes_sha256": file_sha256(train_outcomes),
        **gate,
    }


def evaluate_dev_loaded(data: EvaluationData) -> Mapping[str, Any]:
    evidence = _common_evidence(data)
    gate = decide_dev_gate(evidence)
    return {
        "accessed": True,
        "load_count": 1,
        "evaluation_count": 1,
        "baseline_score": evidence["baseline_score"],
        "candidate_score": evidence["candidate_score_detail"],
        "candidate_score_threshold_exclusive": str(DEV_COMPARATOR),
        "balanced_premium_exact_x1": evidence["balanced_premium_exact_x1"],
        "fast_model_counts_exact_x1": evidence["fast_model_counts_exact_x1"],
        "candidate_fast_predicted_spend_lte_x1": evidence["candidate_fast_predicted_spend_lte_x1"],
        "fast_plan": evidence["fast_plan"],
        "input_path": _repo_path(data.input_path),
        "input_sha256": file_sha256(data.input_path),
        "outcomes_path": _repo_path(data.outcomes_path),
        "outcomes_sha256": file_sha256(data.outcomes_path),
        **gate,
    }


def _risk_data(data: EvaluationData) -> risk_validation.SplitData:
    return risk_validation.SplitData(
        split=data.inputs.split,
        input_path=data.input_path,
        outcomes_path=data.outcomes_path,
        policy=data.policy,
        inputs=data.inputs,
        costs=data.costs,
        scores=data.scores,
        families=data.families,
    )


def _candidate_risk_runner(
    data: risk_validation.SplitData,
) -> risk_validation.PolicyRunner:
    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    residual_artifact = load_residual_artifact()
    predictions = tuple(
        safe_margin.predict_batch(data.inputs.episodes, artifact, data.policy)
    )
    residuals = predict_residuals(data.inputs.episodes, residual_artifact)

    def plan(indices: Sequence[int], tier: str) -> Tuple[str, ...]:
        subset = tuple(predictions[index] for index in indices)
        if tier == "fast":
            subset_residuals = tuple(residuals[index] for index in indices)
            return matched_fast_plan(subset, subset_residuals, data.policy).selected
        return safe_margin.plan_selection(subset, data.policy, tier)[0]

    return risk_validation.PolicyRunner(
        name=EXPERIMENT_ID,
        artifact_path=ARTIFACT_PATH,
        artifact_sha256=EXPECTED_ARTIFACT_SHA256,
        plan=plan,
    )


def evaluate_safety_loaded(data: EvaluationData) -> Mapping[str, Any]:
    measured = _risk_data(data)
    indices = bootstrap_indices(
        measured.num_episodes,
        risk_validation.DEFAULT_RESAMPLES,
        risk_validation.DEFAULT_SEED,
    )
    reference = risk_validation.evaluate_runner(
        measured, risk_validation.safe_margin_runner(measured), indices
    )
    candidate = risk_validation.evaluate_runner(
        measured, _candidate_risk_runner(measured), indices
    )
    return {
        "accessed": True,
        "evaluation_count": 1,
        "resamples": risk_validation.DEFAULT_RESAMPLES,
        "seed": risk_validation.DEFAULT_SEED,
        "reference": reference,
        "candidate": candidate,
        **decide_safety_gate(reference, candidate),
    }


def _round(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentError("non-finite report value")
        return round(value, REPORT_DIGITS)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def run_experiment(
    *,
    train_input: Path = DEFAULT_TRAIN_INPUT,
    train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES,
    dev_input: Path = DEFAULT_DEV_INPUT,
    dev_outcomes: Path = DEFAULT_DEV_OUTCOMES,
    report_path: Path = DEFAULT_REPORT,
) -> Mapping[str, Any]:
    protocol = verify_protocol()
    report: Dict[str, Any] = {
        "base_commit": BASE_COMMIT,
        "experiment_id": EXPERIMENT_ID,
        "report_type": REPORT_TYPE,
        "protocol_path": _repo_path(PROTOCOL_PATH),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_version": protocol["version"],
        "artifact_path": _repo_path(ARTIFACT_PATH),
        "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
        "candidate": {
            "efficiency_buckets_per_octave": 3,
            "scope": "fast light->ax31 only",
            "tuning_sweep": False,
        },
        "production_default": {
            "changed": False,
            "efficiency_buckets_per_octave": safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE,
        },
        "decision": {
            "eligible": False,
            "reason": "Train gate not evaluated",
            "submission_default": "safe-margin",
        },
        "dev": {
            "accessed": False, "load_count": 0, "evaluation_count": 0,
            "gate_passed": False,
        },
        "safety": {
            "accessed": False, "evaluation_count": 0,
            "gate_passed": False, "resamples": 0,
        },
    }
    report["train"] = evaluate_train(train_input, train_outcomes)
    if not report["train"]["gate_passed"]:
        report["decision"]["reason"] = (
            "Train gate failed; Dev and safety remained closed"
        )
    else:
        dev_data = _load_split_once("dev", dev_input, dev_outcomes)
        report["dev"] = evaluate_dev_loaded(dev_data)
        if not report["dev"]["gate_passed"]:
            report["decision"]["reason"] = (
                "strict Public Dev gate failed; safety remained closed"
            )
        else:
            report["safety"] = evaluate_safety_loaded(dev_data)
            if report["safety"]["gate_passed"]:
                report["decision"] = {
                    "eligible": True,
                    "reason": "all frozen gates passed; eligible for later owner consideration",
                    "submission_default": "safe-margin",
                }
            else:
                report["decision"]["reason"] = "5,000-resample safety gate failed"
    rounded = _round(report)
    _atomic_json(report_path, rounded)
    return rounded


def reemit_existing_report(report_path: Path = DEFAULT_REPORT) -> Mapping[str, Any]:
    verify_protocol()
    try:
        value = json.loads(
            report_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ExperimentError(f"cannot re-emit malformed terminal report: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("report_type") != REPORT_TYPE
        or value.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or value.get("artifact_sha256") != EXPECTED_ARTIFACT_SHA256
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("decision", {}).get("submission_default") != "safe-margin"
    ):
        raise ExperimentError("terminal report identity mismatch")
    _atomic_json(report_path, value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--train-outcomes", type=Path, default=DEFAULT_TRAIN_OUTCOMES)
    parser.add_argument("--dev-input", type=Path, default=DEFAULT_DEV_INPUT)
    parser.add_argument("--dev-outcomes", type=Path, default=DEFAULT_DEV_OUTCOMES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reemit-existing", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.reemit_existing:
            reemit_existing_report(args.report)
            print(f"Re-emitted byte-stable report: {_repo_path(args.report)}")
            return 0
        report = run_experiment(
            train_input=args.train_input,
            train_outcomes=args.train_outcomes,
            dev_input=args.dev_input,
            dev_outcomes=args.dev_outcomes,
            report_path=args.report,
        )
    except (ExperimentError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Train={'PASS' if report['train']['gate_passed'] else 'FAIL'}; "
        f"Dev evaluations={report['dev']['evaluation_count']}; "
        f"safety evaluations={report['safety']['evaluation_count']}; "
        f"eligible={report['decision']['eligible']}"
    )
    return 0 if report["decision"]["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
