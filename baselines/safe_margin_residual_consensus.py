# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Train-frozen residual reordering experiment over the safe-margin router.

Only Fast and Balanced ``ax31-light -> ax31`` groups that pass every existing
safe-margin guard can move.  The candidate spends no more conservative
increment than safe-margin selected on the same batch.  Premium delegates to
safe-margin exactly.  This module is an experiment runtime and is deliberately
not wired into the submitted container.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ossp_router.protocol import (
    MODEL_IDS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)

_BASELINES = Path(__file__).resolve().parent
_ROOT = _BASELINES.parent
for _entry in (str(_BASELINES),):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import representation_features  # noqa: E402
import safe_margin  # noqa: E402


STRATEGY_ID = "safe-margin-residual-consensus-v1"
DEFAULT_ARTIFACT_PATH = _BASELINES / "safe-margin-residual-consensus-public.v1.json"
DEFAULT_PROTOCOL_PATH = (
    _ROOT / "configs/safe-margin-residual-consensus-protocol.v1.json"
)
EXCHANGE_TIERS = ("fast", "balanced")
RESIDUAL_CLIP = (-0.1, 0.1)


class ResidualConsensusError(ValueError):
    """Raised when the frozen candidate artifact is invalid."""


@dataclass(frozen=True)
class ResidualArtifact:
    protocol_sha256: str
    policy_id: str
    policy_digest: str
    strength: float
    feature_names: Tuple[str, ...]
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    intercept: float
    coefficients: Tuple[float, ...]


@dataclass(frozen=True)
class ResidualPlan:
    submission: Submission
    predicted_budget_ratio: float
    matched_safe_margin_ratio: float
    model_counts: Mapping[str, int]
    fallback_used: bool = False


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_tuple(value: object, name: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ResidualConsensusError(f"{name} length must be {length}")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ResidualConsensusError(f"{name} must contain finite numbers")
    return result


def load_artifact(
    path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> ResidualArtifact:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidualConsensusError(f"candidate artifact unavailable: {exc}") from exc
    if not isinstance(value, dict) or value.get("strategy_id") != STRATEGY_ID:
        raise ResidualConsensusError("candidate artifact strategy_id mismatch")
    expected_names = tuple(
        representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES
    )
    names = value.get("feature_names")
    if not isinstance(names, list) or tuple(names) != expected_names:
        raise ResidualConsensusError("candidate artifact feature contract mismatch")
    protocol_digest = file_sha256(protocol_path)
    if value.get("protocol_sha256") != protocol_digest:
        raise ResidualConsensusError("candidate artifact protocol hash mismatch")
    strength = float(value.get("strength", float("nan")))
    if strength not in (0.25, 0.5, 1.0):
        raise ResidualConsensusError("candidate strength is outside the frozen grid")
    feature_count = len(expected_names)
    scale = _finite_tuple(value.get("feature_scale"), "feature_scale", feature_count)
    if any(item <= 0.0 for item in scale):
        raise ResidualConsensusError("feature_scale must be positive")
    intercept = float(value.get("intercept", float("nan")))
    if not math.isfinite(intercept):
        raise ResidualConsensusError("intercept must be finite")
    return ResidualArtifact(
        protocol_sha256=protocol_digest,
        policy_id=str(value.get("policy_id", "")),
        policy_digest=str(value.get("policy_sha256", "")),
        strength=strength,
        feature_names=expected_names,
        feature_mean=_finite_tuple(
            value.get("feature_mean"), "feature_mean", feature_count
        ),
        feature_scale=scale,
        intercept=intercept,
        coefficients=_finite_tuple(
            value.get("coefficients"), "coefficients", feature_count
        ),
    )


def predict_residual(episode: Episode, artifact: ResidualArtifact) -> float:
    """Predict a bounded secondary residual from role/content only."""

    vector = representation_features.expanded_structural_vector(episode)
    value = artifact.intercept + math.fsum(
        coefficient * ((feature - mean) / scale)
        for feature, mean, scale, coefficient in zip(
            vector,
            artifact.feature_mean,
            artifact.feature_scale,
            artifact.coefficients,
        )
    )
    return min(RESIDUAL_CLIP[1], max(RESIDUAL_CLIP[0], value))


def predict_residuals(
    episodes: Sequence[Episode], artifact: ResidualArtifact
) -> Tuple[float, ...]:
    return tuple(predict_residual(episode, artifact) for episode in episodes)


def _eligible_groups(
    predictions: Sequence[safe_margin.EpisodePrediction],
    tier: str,
) -> Tuple[Dict[Tuple[int, ...], List[int]], float, float]:
    """Reconstruct the exact safe-margin ax31 eligibility/group contract."""

    config = safe_margin.TIER_PLAN_CONFIGS[tier]
    light_id, ax31_id, _think_id = MODEL_IDS
    light_total = math.fsum(item.costs[light_id] for item in predictions)
    mean_light = light_total / len(predictions)
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for index, prediction in enumerate(predictions):
        gain = prediction.scores[ax31_id] - prediction.scores[light_id]
        increment = prediction.costs[ax31_id] - prediction.costs[light_id]
        step_ratio = prediction.costs[ax31_id] / prediction.costs[light_id]
        step_load = increment / mean_light
        if (
            gain < config.ax31_min_gain
            or increment <= 0.0
            or step_ratio > config.ax31_max_step_ratio
            or step_load > config.ax31_max_step_load
        ):
            continue
        safe_key = (safe_margin._efficiency_bucket(gain / step_load),) + (
            prediction.signature
        )
        groups.setdefault(safe_key, []).append(index)
    return groups, light_total, mean_light


def plan_selection(
    predictions: Sequence[safe_margin.EpisodePrediction],
    residuals: Sequence[float],
    policy: RoutingPolicy,
    tier: str,
    strength: float,
) -> Tuple[Tuple[str, ...], float, float]:
    """Exchange safe-margin-eligible Fast/Balanced groups at matched spend."""

    if len(predictions) != len(residuals) or not predictions:
        raise ValueError("predictions and residuals must be non-empty and aligned")
    safe_selected, safe_ratio, _stages = safe_margin.plan_selection(
        predictions, policy, tier
    )
    if tier not in EXCHANGE_TIERS:
        return safe_selected, safe_ratio, safe_ratio
    if strength not in (0.25, 0.5, 1.0):
        raise ValueError("strength is outside the frozen protocol grid")

    light_id, ax31_id, _think_id = MODEL_IDS
    groups, light_total, mean_light = _eligible_groups(predictions, tier)
    matched_total = math.fsum(
        prediction.costs[model_id]
        for prediction, model_id in zip(predictions, safe_selected)
    )
    selected = [light_id] * len(predictions)
    total = light_total
    priorities = []
    for safe_key, members in groups.items():
        corrected_gain = math.fsum(
            (
                predictions[index].scores[ax31_id]
                - predictions[index].scores[light_id]
                + strength
                * min(RESIDUAL_CLIP[1], max(RESIDUAL_CLIP[0], residuals[index]))
            )
            for index in members
        )
        load = math.fsum(
            (
                predictions[index].costs[ax31_id]
                - predictions[index].costs[light_id]
            )
            / mean_light
            for index in members
        )
        priorities.append((-(corrected_gain / load), safe_key, members))
    for _negative_priority, _safe_key, members in sorted(priorities):
        delta = math.fsum(
            predictions[index].costs[ax31_id]
            - predictions[index].costs[light_id]
            for index in members
        )
        if total + delta > matched_total + 1e-12:
            continue
        total += delta
        for index in members:
            selected[index] = ax31_id
    if total > matched_total + 1e-9:  # pragma: no cover - defensive
        return safe_selected, safe_ratio, safe_ratio
    return tuple(selected), total / light_total, matched_total / light_total


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    base_artifact: safe_margin.HashRegexArtifact,
    residual_artifact: ResidualArtifact,
    tier: str,
) -> ResidualPlan:
    if residual_artifact.policy_id != policy.policy_id:
        raise ProtocolError("candidate artifact policy_id mismatch")
    if residual_artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("candidate artifact policy hash mismatch")
    predictions = safe_margin.predict_batch(inputs.episodes, base_artifact, policy)
    residuals = predict_residuals(inputs.episodes, residual_artifact)
    selected, ratio, matched_ratio = plan_selection(
        predictions, residuals, policy, tier, residual_artifact.strength
    )
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return ResidualPlan(
        submission=parse_submission(submission_to_dict(submission)),
        predicted_budget_ratio=ratio,
        matched_safe_margin_ratio=matched_ratio,
        model_counts={
            model_id: selected.count(model_id) for model_id in MODEL_IDS
        },
    )


def make_submission_or_safe_margin(
    inputs: InputBatch,
    policy: RoutingPolicy,
    base_artifact: safe_margin.HashRegexArtifact,
    tier: str,
    *,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> ResidualPlan:
    """Fail closed to safe-margin when the experimental artifact is unusable."""

    try:
        artifact = load_artifact(artifact_path, protocol_path=protocol_path)
        return make_submission(inputs, policy, base_artifact, artifact, tier)
    except (OSError, ValueError, ProtocolError, json.JSONDecodeError):
        safe = safe_margin.make_safe_margin_submission(
            inputs, policy, base_artifact, tier
        )
        return ResidualPlan(
            submission=safe.submission,
            predicted_budget_ratio=safe.predicted_budget_ratio,
            matched_safe_margin_ratio=safe.predicted_budget_ratio,
            model_counts=safe.model_counts,
            fallback_used=True,
        )
