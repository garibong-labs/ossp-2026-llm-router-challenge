# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed Premium think-loss veto layered on unchanged safe-margin.

The candidate can only replace complete Premium ``axk1-think`` content groups
with ``ax31``.  Invalid, missing, or rejected artifacts return the original
safe-margin plan byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ossp_router.protocol import (
    MODEL_IDS,
    Decision,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    policy_sha256,
)

_BASELINES = str(Path(__file__).resolve().parent)
if _BASELINES not in sys.path:
    sys.path.insert(0, _BASELINES)

import representation_features  # noqa: E402
import safe_margin  # noqa: E402


ARTIFACT_TYPE = "safe-margin-think-loss-veto-v1"
FEATURE_VERSION = "B-expanded-structural-plus-safe-margin-v1"
PROTOCOL_SHA256 = "651445bf7ec35d6779a08025c613879bcc606ec539428f95119c1b8d43a1131c"
BASE_COMMIT = "3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9"
STRUCTURAL_FEATURE_NAMES = tuple(
    representation_features.EXPANDED_STRUCTURAL_FEATURE_NAMES
)
RUNTIME_FEATURE_NAMES = (
    "safe_margin_predicted_think_minus_ax31_gain",
    "safe_margin_predicted_think_minus_ax31_conservative_cost_increment",
    "safe_margin_predicted_think_gain_per_step_load_efficiency_bucket",
)
FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + RUNTIME_FEATURE_NAMES
STRUCTURAL_FEATURE_COUNT = len(STRUCTURAL_FEATURE_NAMES)
DEFAULT_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/safe-margin-think-loss-veto/artifact.v1.json"
)


@dataclass(frozen=True)
class VetoHead:
    feature_indices: Tuple[int, ...]
    mean: Tuple[float, ...]
    scale: Tuple[float, ...]
    intercept: float
    coefficients: Tuple[float, ...]
    upper_residual: float

    def predict(self, values: Sequence[float]) -> float:
        standardized = (
            (values[index] - mean) / scale
            for index, mean, scale in zip(
                self.feature_indices, self.mean, self.scale
            )
        )
        return self.intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(self.coefficients, standardized)
        )


@dataclass(frozen=True)
class VetoArtifact:
    heads: Tuple[VetoHead, ...]
    minimum_negative_votes: int
    policy_id: str
    policy_digest: str


@dataclass(frozen=True)
class VetoPlan:
    submission: Submission
    baseline: safe_margin.SafeMarginPlan
    artifact_valid: bool
    groups_considered: int
    groups_vetoed: int
    episodes_vetoed: int


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be a finite number")
    return result


def _vector(value: Any, length: int, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label} must contain {length} numbers")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def parse_artifact(value: Any) -> VetoArtifact:
    if not isinstance(value, dict):
        raise ProtocolError("veto artifact must be an object")
    required = {
        "artifact_type", "schema_version", "protocol_sha256", "base_commit",
        "feature_version", "feature_names", "heads", "minimum_negative_votes",
        "policy_id", "policy_sha256", "training_data_sha256",
    }
    if set(value) != required:
        raise ProtocolError("veto artifact fields do not match the frozen schema")
    names = list(FEATURE_NAMES)
    if (
        value["artifact_type"] != ARTIFACT_TYPE
        or value["schema_version"] != 1
        or value["protocol_sha256"] != PROTOCOL_SHA256
        or value["base_commit"] != BASE_COMMIT
        or value["feature_version"] != FEATURE_VERSION
        or value["feature_names"] != names
    ):
        raise ProtocolError("veto artifact identity does not match the frozen protocol")
    threshold = value["minimum_negative_votes"]
    if threshold not in (7, 8, 9):
        raise ProtocolError("artifact has no adopted frozen consensus threshold")
    raw_heads = value["heads"]
    if not isinstance(raw_heads, list) or len(raw_heads) != 9:
        raise ProtocolError("artifact must contain exactly nine heads")
    heads = []
    for head_index, raw in enumerate(raw_heads):
        if not isinstance(raw, dict) or set(raw) != {
            "head", "feature_indices", "mean", "scale", "intercept",
            "coefficients", "upper_residual",
        }:
            raise ProtocolError("head fields do not match the frozen schema")
        indices = raw["feature_indices"]
        expected = [
            index for index in range(STRUCTURAL_FEATURE_COUNT)
            if index % 9 != head_index
        ] + list(range(STRUCTURAL_FEATURE_COUNT, len(FEATURE_NAMES)))
        if raw["head"] != head_index or indices != expected:
            raise ProtocolError("head feature partition is not frozen modulo-9 partition")
        length = len(indices)
        scale = _vector(raw["scale"], length, f"heads[{head_index}].scale")
        if any(item <= 0 for item in scale):
            raise ProtocolError("head feature scales must be positive")
        heads.append(VetoHead(
            feature_indices=tuple(indices),
            mean=_vector(raw["mean"], length, f"heads[{head_index}].mean"),
            scale=scale,
            intercept=_number(raw["intercept"], f"heads[{head_index}].intercept"),
            coefficients=_vector(raw["coefficients"], length, f"heads[{head_index}].coefficients"),
            upper_residual=_number(raw["upper_residual"], f"heads[{head_index}].upper_residual"),
        ))
    training_hash = value["training_data_sha256"]
    if not isinstance(training_hash, str) or len(training_hash) != 64:
        raise ProtocolError("training data digest is malformed")
    if not isinstance(value["policy_id"], str) or not isinstance(value["policy_sha256"], str):
        raise ProtocolError("artifact policy identity is malformed")
    return VetoArtifact(tuple(heads), threshold, value["policy_id"], value["policy_sha256"])


def load_artifact(path: Path = DEFAULT_ARTIFACT_PATH) -> VetoArtifact:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot load veto artifact: {exc}") from exc
    return parse_artifact(raw)


def content_group_keys(
    predictions: Sequence[safe_margin.EpisodePrediction],
) -> Tuple[Tuple[int, ...], ...]:
    """Return the unchanged safe-margin think-stage grouping key per row."""

    light_total = math.fsum(item.costs[MODEL_IDS[0]] for item in predictions)
    mean_light = light_total / len(predictions)
    result = []
    for prediction in predictions:
        gain = prediction.scores[MODEL_IDS[2]] - prediction.scores[MODEL_IDS[1]]
        increment = prediction.costs[MODEL_IDS[2]] - prediction.costs[MODEL_IDS[1]]
        efficiency = gain / (increment / mean_light) if increment > 0 else 0.0
        result.append((safe_margin._efficiency_bucket(efficiency),) + prediction.signature)
    return tuple(result)


def feature_vectors(
    inputs: InputBatch,
    predictions: Sequence[safe_margin.EpisodePrediction],
) -> Tuple[Tuple[float, ...], ...]:
    """Build frozen representation B plus the three safe-margin additions."""

    if len(inputs.episodes) != len(predictions):
        raise ValueError("episode and safe-margin prediction counts differ")
    if not predictions:
        return ()
    mean_light = math.fsum(
        item.costs[MODEL_IDS[0]] for item in predictions
    ) / len(predictions)
    vectors = []
    for episode, prediction in zip(inputs.episodes, predictions):
        gain = prediction.scores[MODEL_IDS[2]] - prediction.scores[MODEL_IDS[1]]
        increment = prediction.costs[MODEL_IDS[2]] - prediction.costs[MODEL_IDS[1]]
        step_load = increment / mean_light
        efficiency = gain / step_load if increment > 0 else 0.0
        additions = (
            float(gain),
            float(increment),
            float(safe_margin._efficiency_bucket(efficiency)),
        )
        vectors.append(tuple(
            representation_features.expanded_structural_vector(episode)
        ) + additions)
    return tuple(vectors)


def apply_veto(
    inputs: InputBatch,
    baseline: safe_margin.SafeMarginPlan,
    predictions: Sequence[safe_margin.EpisodePrediction],
    artifact: VetoArtifact,
) -> VetoPlan:
    if baseline.submission.tier != "premium":
        return VetoPlan(baseline.submission, baseline, True, 0, 0, 0)
    vectors = feature_vectors(inputs, predictions)
    keys = content_group_keys(predictions)
    groups: Dict[Tuple[int, ...], list[int]] = {}
    selected = [decision.model_id for decision in baseline.submission.decisions]
    for index, model_id in enumerate(selected):
        if model_id == MODEL_IDS[2]:
            groups.setdefault(keys[index], []).append(index)
    vetoed = 0
    episodes_vetoed = 0
    for members in groups.values():
        votes = 0
        for head in artifact.heads:
            predicted = math.fsum(head.predict(vectors[index]) for index in members) / len(members)
            if predicted + head.upper_residual < 0.0:
                votes += 1
        if votes >= artifact.minimum_negative_votes:
            vetoed += 1
            episodes_vetoed += len(members)
            for index in members:
                selected[index] = MODEL_IDS[1]
    submission = Submission(
        schema_version=baseline.submission.schema_version,
        challenge_id=baseline.submission.challenge_id,
        policy_id=baseline.submission.policy_id,
        split=baseline.submission.split,
        tier=baseline.submission.tier,
        decisions=tuple(
            Decision(episode.episode_id, model_id)
            for episode, model_id in zip(inputs.episodes, selected)
        ),
    )
    return VetoPlan(submission, baseline, True, len(groups), vetoed, episodes_vetoed)


def make_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    safe_artifact: safe_margin.HashRegexArtifact,
    tier: str,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
) -> VetoPlan:
    """Build the candidate or return unchanged safe-margin on any artifact fault."""

    baseline = safe_margin.make_safe_margin_submission(inputs, policy, safe_artifact, tier)
    try:
        artifact = load_artifact(artifact_path)
        if artifact.policy_id != policy.policy_id or artifact.policy_digest != policy_sha256(policy):
            raise ProtocolError("veto artifact policy does not match runtime policy")
        predictions = safe_margin.predict_batch(inputs.episodes, safe_artifact, policy)
        return apply_veto(inputs, baseline, predictions, artifact)
    except (OSError, ValueError, ProtocolError):
        return VetoPlan(baseline.submission, baseline, False, 0, 0, 0)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
