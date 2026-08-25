# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Risk-calibrated v2 prompt-only router (candidate policy, stdlib runtime).

The policy predicts the two upgrade steps *directly* instead of differencing
per-model score heads:

* ``ax31-light -> ax31`` and ``ax31 -> axk1-think`` each get an expected-gain
  head plus a calibrated probability-of-positive-gain head;
* per-model log-cost heads keep the public input-token-rate floor and the
  monotone model-cost ordering of the safe-margin policy;
* a split-conformal upper factor per model turns the mean cost estimate into
  an upper-tail estimate with measured held-out coverage. The upper estimate
  drives every per-episode guard, so an episode whose upgrade *could* be
  expensive is blocked even when its mean estimate looks cheap. An artifact
  whose measured coverage misses its declared level fails validation and
  therefore can never authorize additional budget use.

Upgrades are ranked by expected incremental quality per risk-adjusted
incremental cost and allocated under the same hard guards as the safe-margin
router (per-episode ``max_step_ratio`` expansion guard, per-episode
``max_step_load`` concentration guard, and a bounded Premium think
sub-budget). Content-identical episodes share an exact prediction/signature
group key and are promoted or kept together, so no episode ID, split label or
input position can break a tie.

This module is a *candidate*: the submitted container still runs the
safe-margin policy because the measured champion gates in
``baselines/risk-validation-report.v2.json`` were not met. When this policy is
invoked directly and its artifact fails validation, the router falls back to
the deterministic safe-margin policy instead of guessing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ossp_router.heuristic import write_submission_atomic
from ossp_router.protocol import (
    MODEL_IDS,
    TIERS,
    Decision,
    Episode,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_json,
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)

_BASELINES_DIRECTORY = str(Path(__file__).resolve().parent)
if _BASELINES_DIRECTORY not in sys.path:
    sys.path.insert(0, _BASELINES_DIRECTORY)

from hash_regex import (  # noqa: E402  (sibling baseline module)
    DENSE_FEATURE_NAMES,
    FEATURE_VERSION,
    LinearHead,
    MAX_HASH_BINS,
    MIN_HASH_BINS,
    raw_feature_vector,
)
from safe_margin import content_signature  # noqa: E402  (sibling baseline module)


STRATEGY_ID = "risk-calibrated-v2"
ARTIFACT_TYPE = "ossp-risk-calibrated-linear-v2"

#: Bundled candidate artifact next to this module, mirroring the safe-margin
#: bundling convention so a container could resolve it without arguments.
DEFAULT_ARTIFACT_PATH = Path(_BASELINES_DIRECTORY) / "risk-calibrated-public.v2.json"

#: The measured split-conformal coverage may fall at most this far below the
#: declared level before the artifact is rejected (and the caller falls back).
COVERAGE_SLACK = 0.03

_STEPS = ("ax31", "axk1-think")


@dataclass(frozen=True)
class PlattCalibration:
    """Monotone logistic map from a raw head score to a probability."""

    scale: float
    offset: float

    def apply(self, value: float) -> float:
        z = self.scale * value + self.offset
        z = min(50.0, max(-50.0, z))
        return 1.0 / (1.0 + math.exp(-z))


@dataclass(frozen=True)
class IsotonicCalibration:
    """Right-continuous step map fitted with pool-adjacent-violators."""

    thresholds: Tuple[float, ...]
    values: Tuple[float, ...]

    def apply(self, value: float) -> float:
        result = self.values[0]
        for threshold, mapped in zip(self.thresholds, self.values):
            if value >= threshold:
                result = mapped
            else:
                break
        return result


@dataclass(frozen=True)
class TierPlanV2:
    """Frozen per-tier safety envelope of the risk-calibrated policy."""

    target_ratio: float
    min_gain: float
    min_probability: float
    max_step_ratio: float
    max_step_load: float
    allow_think: bool
    think_min_gain: float
    think_min_probability: float
    think_max_step_ratio: float
    think_max_step_load: float
    think_budget_share: float


@dataclass(frozen=True)
class RiskCalibratedArtifact:
    hash_bins: int
    feature_mean: Tuple[float, ...]
    feature_scale: Tuple[float, ...]
    gain_heads: Mapping[str, LinearHead]
    gain_probability_heads: Mapping[str, LinearHead]
    gain_probability_calibration: Mapping[str, Any]
    log_cost_heads: Mapping[str, LinearHead]
    cost_upper_factors: Mapping[str, float]
    cost_upper_declared_coverage: float
    cost_upper_measured_coverage: Mapping[str, float]
    tier_plans: Mapping[str, TierPlanV2]
    policy_id: str
    policy_digest: str
    training_summary: Mapping[str, Any]


@dataclass(frozen=True)
class EpisodePredictionV2:
    """Per-episode upgrade predictions plus the content signature."""

    gains: Mapping[str, float]
    probabilities: Mapping[str, float]
    costs: Mapping[str, float]
    upper_costs: Mapping[str, float]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class StageReportV2:
    step: str
    considered: int
    eligible: int
    promoted: int
    groups_considered: int
    groups_promoted: int
    budget: float
    spent_ratio: float


@dataclass(frozen=True)
class RiskCalibratedPlan:
    submission: Submission
    predicted_budget_ratio: float
    target_ratio: float
    model_counts: Mapping[str, int]
    stages: Tuple[StageReportV2, ...]


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{label}은(는) JSON 객체여야 합니다.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise ProtocolError(f"{label} 필드 오류: 누락={missing}, 초과={extra}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ProtocolError(f"{label}은(는) 유한한 숫자여야 합니다.")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label}은(는) 유한한 숫자여야 합니다.")
    return result


def _bounded(value: Any, label: str, minimum: float, maximum: float) -> float:
    result = _number(value, label)
    if not minimum <= result <= maximum:
        raise ProtocolError(f"{label} 값이 허용 범위를 벗어났습니다.")
    return result


def _vector(value: Any, length: int, label: str) -> Tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ProtocolError(f"{label}은(는) 길이 {length}의 배열이어야 합니다.")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _head(value: Any, length: int, label: str) -> LinearHead:
    raw = _object(value, label)
    _exact_keys(raw, ("intercept", "coefficients"), label)
    return LinearHead(
        intercept=_number(raw["intercept"], f"{label}.intercept"),
        coefficients=_vector(raw["coefficients"], length, f"{label}.coefficients"),
    )


def _calibration(value: Any, label: str) -> Any:
    raw = _object(value, label)
    method = raw.get("method")
    if method == "platt":
        _exact_keys(raw, ("method", "scale", "offset"), label)
        scale = _number(raw["scale"], f"{label}.scale")
        if scale <= 0:
            raise ProtocolError(f"{label}.scale은 0보다 커야 합니다.")
        return PlattCalibration(
            scale=scale, offset=_number(raw["offset"], f"{label}.offset")
        )
    if method == "isotonic":
        _exact_keys(raw, ("method", "thresholds", "values"), label)
        thresholds = raw["thresholds"]
        values = raw["values"]
        if (
            not isinstance(thresholds, list)
            or not isinstance(values, list)
            or not thresholds
            or len(thresholds) != len(values)
        ):
            raise ProtocolError(f"{label} 배열 구성이 올바르지 않습니다.")
        parsed_thresholds = tuple(
            _number(item, f"{label}.thresholds[{index}]")
            for index, item in enumerate(thresholds)
        )
        parsed_values = tuple(
            _bounded(item, f"{label}.values[{index}]", 0.0, 1.0)
            for index, item in enumerate(values)
        )
        if list(parsed_thresholds) != sorted(parsed_thresholds) or list(
            parsed_values
        ) != sorted(parsed_values):
            raise ProtocolError(f"{label}은(는) 단조 증가해야 합니다.")
        return IsotonicCalibration(
            thresholds=parsed_thresholds, values=parsed_values
        )
    raise ProtocolError(f"{label}.method가 올바르지 않습니다.")


_TIER_PLAN_FIELDS = (
    "target_ratio",
    "min_gain",
    "min_probability",
    "max_step_ratio",
    "max_step_load",
    "allow_think",
    "think_min_gain",
    "think_min_probability",
    "think_max_step_ratio",
    "think_max_step_load",
    "think_budget_share",
)


def _tier_plan(value: Any, label: str) -> TierPlanV2:
    raw = _object(value, label)
    _exact_keys(raw, _TIER_PLAN_FIELDS, label)
    allow_think = raw["allow_think"]
    if not isinstance(allow_think, bool):
        raise ProtocolError(f"{label}.allow_think는 불리언이어야 합니다.")
    return TierPlanV2(
        target_ratio=_bounded(raw["target_ratio"], f"{label}.target_ratio", 1.0, 8.0),
        min_gain=_bounded(raw["min_gain"], f"{label}.min_gain", 0.0, 1.0),
        min_probability=_bounded(
            raw["min_probability"], f"{label}.min_probability", 0.0, 1.0
        ),
        max_step_ratio=_bounded(
            raw["max_step_ratio"], f"{label}.max_step_ratio", 0.0, 1000.0
        ),
        max_step_load=_bounded(
            raw["max_step_load"], f"{label}.max_step_load", 0.0, 1000.0
        ),
        allow_think=allow_think,
        think_min_gain=_bounded(
            raw["think_min_gain"], f"{label}.think_min_gain", 0.0, 1.0
        ),
        think_min_probability=_bounded(
            raw["think_min_probability"], f"{label}.think_min_probability", 0.0, 1.0
        ),
        think_max_step_ratio=_bounded(
            raw["think_max_step_ratio"], f"{label}.think_max_step_ratio", 0.0, 1000.0
        ),
        think_max_step_load=_bounded(
            raw["think_max_step_load"], f"{label}.think_max_step_load", 0.0, 1000.0
        ),
        think_budget_share=_bounded(
            raw["think_budget_share"], f"{label}.think_budget_share", 0.0, 1.0
        ),
    )


def parse_artifact(value: Any) -> RiskCalibratedArtifact:
    root = _object(value, "artifact")
    expected = (
        "artifact_type",
        "schema_version",
        "feature_version",
        "hash_algorithm",
        "hash_bins",
        "dense_feature_names",
        "model_ids",
        "policy_id",
        "policy_sha256",
        "feature_mean",
        "feature_scale",
        "gain_heads",
        "gain_probability_heads",
        "gain_probability_calibration",
        "log_cost_heads",
        "cost_upper",
        "tier_plans",
        "training_summary",
    )
    _exact_keys(root, expected, "artifact")
    if root["artifact_type"] != ARTIFACT_TYPE:
        raise ProtocolError("지원하지 않는 risk-calibrated artifact_type입니다.")
    if root["schema_version"] != 1 or root["feature_version"] != FEATURE_VERSION:
        raise ProtocolError("지원하지 않는 risk-calibrated artifact 버전입니다.")
    if root["hash_algorithm"] != "fnv1a64-signed-word-1-2":
        raise ProtocolError("지원하지 않는 feature hash 방식입니다.")
    hash_bins = root["hash_bins"]
    if (
        isinstance(hash_bins, bool)
        or not isinstance(hash_bins, int)
        or not MIN_HASH_BINS <= hash_bins <= MAX_HASH_BINS
        or hash_bins & (hash_bins - 1)
    ):
        raise ProtocolError("artifact.hash_bins는 허용 범위의 2의 거듭제곱이어야 합니다.")
    if root["dense_feature_names"] != list(DENSE_FEATURE_NAMES):
        raise ProtocolError("dense feature 정의가 현재 런타임과 다릅니다.")
    if root["model_ids"] != list(MODEL_IDS):
        raise ProtocolError("artifact.model_ids가 공개 정책 모델과 다릅니다.")
    length = len(DENSE_FEATURE_NAMES) + hash_bins
    mean = _vector(root["feature_mean"], length, "artifact.feature_mean")
    scale = _vector(root["feature_scale"], length, "artifact.feature_scale")
    if any(item <= 0 for item in scale):
        raise ProtocolError("artifact.feature_scale은 모두 0보다 커야 합니다.")

    gain_raw = _object(root["gain_heads"], "artifact.gain_heads")
    prob_raw = _object(
        root["gain_probability_heads"], "artifact.gain_probability_heads"
    )
    calibration_raw = _object(
        root["gain_probability_calibration"],
        "artifact.gain_probability_calibration",
    )
    if (
        set(gain_raw) != set(_STEPS)
        or set(prob_raw) != set(_STEPS)
        or set(calibration_raw) != set(_STEPS)
    ):
        raise ProtocolError("upgrade head의 단계 집합이 올바르지 않습니다.")
    cost_raw = _object(root["log_cost_heads"], "artifact.log_cost_heads")
    if set(cost_raw) != set(MODEL_IDS):
        raise ProtocolError("artifact 비용 head의 모델 집합이 올바르지 않습니다.")

    upper_raw = _object(root["cost_upper"], "artifact.cost_upper")
    _exact_keys(
        upper_raw,
        ("declared_coverage", "factors", "measured_coverage"),
        "artifact.cost_upper",
    )
    declared = _bounded(
        upper_raw["declared_coverage"],
        "artifact.cost_upper.declared_coverage",
        0.5,
        1.0,
    )
    factors_raw = _object(upper_raw["factors"], "artifact.cost_upper.factors")
    measured_raw = _object(
        upper_raw["measured_coverage"], "artifact.cost_upper.measured_coverage"
    )
    if set(factors_raw) != set(MODEL_IDS) or set(measured_raw) != set(MODEL_IDS):
        raise ProtocolError("cost_upper의 모델 집합이 올바르지 않습니다.")
    factors = {
        model_id: _bounded(
            factors_raw[model_id],
            f"artifact.cost_upper.factors.{model_id}",
            1.0,
            1000.0,
        )
        for model_id in MODEL_IDS
    }
    measured = {
        model_id: _bounded(
            measured_raw[model_id],
            f"artifact.cost_upper.measured_coverage.{model_id}",
            0.0,
            1.0,
        )
        for model_id in MODEL_IDS
    }
    for model_id in MODEL_IDS:
        if measured[model_id] < declared - COVERAGE_SLACK:
            # A cost model that misses its declared coverage gate must not
            # authorize additional budget use.
            raise ProtocolError(
                f"cost_upper.measured_coverage.{model_id}가 선언된 "
                f"{declared}보다 낮습니다."
            )

    plans_raw = _object(root["tier_plans"], "artifact.tier_plans")
    if set(plans_raw) != set(TIERS):
        raise ProtocolError("artifact.tier_plans의 tier 집합이 올바르지 않습니다.")
    policy_id = root["policy_id"]
    policy_digest = root["policy_sha256"]
    if not isinstance(policy_id, str) or not policy_id:
        raise ProtocolError("artifact.policy_id가 올바르지 않습니다.")
    if (
        not isinstance(policy_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", policy_digest) is None
    ):
        raise ProtocolError("artifact.policy_sha256가 올바르지 않습니다.")
    return RiskCalibratedArtifact(
        hash_bins=hash_bins,
        feature_mean=mean,
        feature_scale=scale,
        gain_heads={
            step: _head(gain_raw[step], length, f"gain_heads.{step}")
            for step in _STEPS
        },
        gain_probability_heads={
            step: _head(prob_raw[step], length, f"gain_probability_heads.{step}")
            for step in _STEPS
        },
        gain_probability_calibration={
            step: _calibration(
                calibration_raw[step], f"gain_probability_calibration.{step}"
            )
            for step in _STEPS
        },
        log_cost_heads={
            model_id: _head(cost_raw[model_id], length, f"log_cost_heads.{model_id}")
            for model_id in MODEL_IDS
        },
        cost_upper_factors=factors,
        cost_upper_declared_coverage=declared,
        cost_upper_measured_coverage=measured,
        tier_plans={
            tier: _tier_plan(plans_raw[tier], f"tier_plans.{tier}") for tier in TIERS
        },
        policy_id=policy_id,
        policy_digest=policy_digest,
        training_summary=dict(_object(root["training_summary"], "training_summary")),
    )


def load_artifact(path: Path) -> RiskCalibratedArtifact:
    return parse_artifact(load_json(path))


def _linear(head: LinearHead, values: Sequence[float]) -> float:
    return head.intercept + math.fsum(
        coefficient * value
        for coefficient, value in zip(head.coefficients, values)
    )


def _rate_ratio(policy: RoutingPolicy, model_id: str) -> float:
    light = float(policy.models[policy.light_model_id].input_token_rate)
    if light <= 0:
        raise ProtocolError("light 모델의 input_token_rate는 0보다 커야 합니다.")
    return float(policy.models[model_id].input_token_rate) / light


def _floored_monotone(
    raw_costs: Mapping[str, float], policy: RoutingPolicy
) -> Dict[str, float]:
    """Floor learned costs at the public rate ratio and force the ladder order."""

    light = raw_costs[MODEL_IDS[0]]
    if not math.isfinite(light) or light <= 0:
        raise ValueError("light 모델의 예측 비용은 0보다 큰 유한값이어야 합니다.")
    result: Dict[str, float] = {}
    previous = 0.0
    for model_id in MODEL_IDS:
        floored = max(raw_costs[model_id], light * _rate_ratio(policy, model_id))
        floored = max(floored, previous * (1.0 + 1e-12))
        result[model_id] = floored
        previous = floored
    return result


def predict_episode(
    episode: Episode, artifact: RiskCalibratedArtifact, policy: RoutingPolicy
) -> EpisodePredictionV2:
    raw = raw_feature_vector(episode, artifact.hash_bins)
    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            raw, artifact.feature_mean, artifact.feature_scale
        )
    )
    gains = {
        step: min(1.0, max(-1.0, _linear(artifact.gain_heads[step], standardized)))
        for step in _STEPS
    }
    probabilities = {
        step: min(
            1.0,
            max(
                0.0,
                artifact.gain_probability_calibration[step].apply(
                    _linear(artifact.gain_probability_heads[step], standardized)
                ),
            ),
        )
        for step in _STEPS
    }
    raw_costs = {
        model_id: math.exp(
            min(
                50.0,
                max(
                    -50.0,
                    _linear(artifact.log_cost_heads[model_id], standardized),
                ),
            )
        )
        for model_id in MODEL_IDS
    }
    costs = _floored_monotone(raw_costs, policy)
    upper = _floored_monotone(
        {
            model_id: costs[model_id] * artifact.cost_upper_factors[model_id]
            for model_id in MODEL_IDS
        },
        policy,
    )
    return EpisodePredictionV2(
        gains=gains,
        probabilities=probabilities,
        costs=costs,
        upper_costs=upper,
        signature=content_signature(raw),
    )


def predict_batch(
    episodes: Sequence[Episode],
    artifact: RiskCalibratedArtifact,
    policy: RoutingPolicy,
) -> Tuple[EpisodePredictionV2, ...]:
    return tuple(
        predict_episode(episode, artifact, policy) for episode in episodes
    )


def _run_stage(
    *,
    step_from: str,
    step_to: str,
    step_key: str,
    selected: List[str],
    predictions: Sequence[EpisodePredictionV2],
    light_total: float,
    mean_light: float,
    current_total: float,
    budget: float,
    min_gain: float,
    min_probability: float,
    max_step_ratio: float,
    max_step_load: float,
) -> Tuple[float, StageReportV2]:
    """Promote exact prediction/signature groups by risk-adjusted efficiency."""

    groups: Dict[Tuple[Any, ...], List[int]] = {}
    considered = 0
    eligible = 0
    light_id = MODEL_IDS[0]
    for index, prediction in enumerate(predictions):
        if selected[index] != step_from:
            continue
        considered += 1
        gain = prediction.gains[step_key]
        probability = prediction.probabilities[step_key]
        # The budget is spent in *risk-adjusted* units: the upper-tail cost of
        # the model being stepped to, minus the mean cost of the model being
        # stepped from. Mean-only spending would systematically admit the
        # episodes whose costs the head underestimates (a winner's curse that
        # measurably overshoots on the split the head was not fitted on).
        risk_increment = (
            prediction.upper_costs[step_to] - prediction.costs[step_from]
        )
        step_ratio = prediction.upper_costs[step_to] / prediction.costs[light_id]
        step_load = risk_increment / mean_light
        if (
            gain < min_gain
            or probability < min_probability
            or risk_increment <= 0
            or step_ratio > max_step_ratio
            or step_load > max_step_load
        ):
            continue
        eligible += 1
        efficiency = gain / step_load
        key = (efficiency, prediction.signature, risk_increment, gain)
        groups.setdefault(key, []).append(index)

    promoted = 0
    groups_promoted = 0
    total = current_total
    # Descending risk-adjusted efficiency, then ascending content signature and
    # planned increment: every key component is derived from prompt content
    # only, so input order can never break a tie.
    for key in sorted(groups, key=lambda item: (-item[0], item[1], item[2], item[3])):
        members = groups[key]
        delta = math.fsum(
            predictions[index].upper_costs[step_to]
            - predictions[index].costs[step_from]
            for index in members
        )
        if total + delta > budget:
            continue
        total += delta
        groups_promoted += 1
        for index in members:
            selected[index] = step_to
            promoted += 1

    return total, StageReportV2(
        step=f"{step_from}->{step_to}",
        considered=considered,
        eligible=eligible,
        promoted=promoted,
        groups_considered=len(groups),
        groups_promoted=groups_promoted,
        budget=budget / light_total,
        spent_ratio=total / light_total,
    )


def plan_selection(
    predictions: Sequence[EpisodePredictionV2],
    policy: RoutingPolicy,
    tier: str,
    plans: Mapping[str, TierPlanV2],
) -> Tuple[Tuple[str, ...], float, Tuple[StageReportV2, ...]]:
    """Return one model per episode plus the planned cost ratio and audit."""

    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if not predictions:
        raise ValueError("예측 배열은 비어 있을 수 없습니다.")
    plan = plans[tier]
    light_id, ax31_id, think_id = MODEL_IDS

    light_total = math.fsum(item.costs[light_id] for item in predictions)
    if light_total <= 0:
        raise ValueError("예측 light 비용 합계는 0보다 커야 합니다.")
    mean_light = light_total / len(predictions)

    budget_multiplier = float(policy.tiers[tier].budget_multiplier)
    target_ratio = min(plan.target_ratio, budget_multiplier)
    full_budget = light_total * target_ratio

    selected: List[str] = [light_id] * len(predictions)
    stages: List[StageReportV2] = []

    total, ax31_stage = _run_stage(
        step_from=light_id,
        step_to=ax31_id,
        step_key="ax31",
        selected=selected,
        predictions=predictions,
        light_total=light_total,
        mean_light=mean_light,
        current_total=light_total,
        budget=full_budget,
        min_gain=plan.min_gain,
        min_probability=plan.min_probability,
        max_step_ratio=plan.max_step_ratio,
        max_step_load=plan.max_step_load,
    )
    stages.append(ax31_stage)

    if plan.allow_think:
        think_allowance = (full_budget - light_total) * plan.think_budget_share
        total, think_stage = _run_stage(
            step_from=ax31_id,
            step_to=think_id,
            step_key="axk1-think",
            selected=selected,
            predictions=predictions,
            light_total=light_total,
            mean_light=mean_light,
            current_total=total,
            budget=min(full_budget, total + think_allowance),
            min_gain=plan.think_min_gain,
            min_probability=plan.think_min_probability,
            max_step_ratio=plan.think_max_step_ratio,
            max_step_load=plan.think_max_step_load,
        )
        stages.append(think_stage)

    if total > full_budget:  # pragma: no cover - stages never exceed the budget
        return (
            tuple(light_id for _item in predictions),
            1.0,
            tuple(stages),
        )
    return tuple(selected), total / light_total, tuple(stages)


def make_risk_calibrated_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: RiskCalibratedArtifact,
    tier: str,
) -> RiskCalibratedPlan:
    """Create one complete v1 submission for a single tier."""

    if inputs.schema_version != policy.schema_version:
        raise ProtocolError("입력과 정책의 schema_version이 일치하지 않습니다.")
    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if artifact.policy_id != policy.policy_id:
        raise ProtocolError("artifact와 정책의 policy_id가 다릅니다.")
    if artifact.policy_digest != policy_sha256(policy):
        raise ProtocolError("artifact와 현재 정책의 SHA-256이 다릅니다.")
    predictions = predict_batch(inputs.episodes, artifact, policy)
    selected, ratio, stages = plan_selection(
        predictions, policy, tier, artifact.tier_plans
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
    counts = {
        model_id: sum(1 for item in selected if item == model_id)
        for model_id in MODEL_IDS
    }
    return RiskCalibratedPlan(
        submission=parse_submission(submission_to_dict(submission)),
        predicted_budget_ratio=ratio,
        target_ratio=artifact.tier_plans[tier].target_ratio,
        model_counts=counts,
        stages=stages,
    )


def _fallback_submission(
    inputs: InputBatch, policy: RoutingPolicy, tier: str
) -> Submission:
    """Deterministic safe-margin fallback when the v2 artifact is unusable."""

    import safe_margin

    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    return safe_margin.make_safe_margin_submission(
        inputs, policy, artifact, tier
    ).submission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk-calibrated-router",
        description="위험 보정 v2 prompt-only 라우터(후보 정책)",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="artifact 오류 시 safe-margin 대체 실행을 비활성화합니다.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inputs = load_input(args.input)
        policy = (
            load_policy(args.policy)
            if args.policy is not None
            else load_bundled_policy()
        )
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    try:
        artifact = load_artifact(args.artifact)
        plan = make_risk_calibrated_submission(inputs, policy, artifact, args.tier)
        submission = plan.submission
        message = (
            f"OK: {args.tier} 제출 파일을 생성했습니다 "
            f"(예측 비용 비율 {plan.predicted_budget_ratio:.6f}, "
            f"안전 목표 {plan.target_ratio:.2f}, "
            + ", ".join(
                f"{model_id}={plan.model_counts[model_id]}"
                for model_id in MODEL_IDS
            )
            + ")."
        )
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        if args.no_fallback:
            print(f"오류: {exc}", file=sys.stderr)
            return 2
        try:
            submission = _fallback_submission(inputs, policy, args.tier)
        except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as inner:
            print(f"오류: {exc} / 대체 실행도 실패: {inner}", file=sys.stderr)
            return 2
        message = (
            f"OK: {args.tier} 제출 파일을 safe-margin 대체 정책으로 "
            f"생성했습니다 (사유: {exc})."
        )
    try:
        write_submission_atomic(args.output, submission)
    except OSError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
