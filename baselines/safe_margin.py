# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Conservative prompt-only router that keeps a deliberate budget margin.

The policy reuses the public hash-regex feature extraction and the bundled
public artifact for per-model quality and cost prediction, but replaces the
Lagrangian batch selection with a conservative hybrid:

* per-model cost is floored by the public policy input-token rate ratio and
  then inflated, so a drifting learned cost head cannot silently understate an
  upgrade;
* upgrades are ranked by predicted marginal quality per incremental cost and
  allocated in content-derived groups, never per episode identity or position;
* every upgrade must clear a quality margin, a per-episode *expansion* guard
  (`max_step_ratio`) and a per-episode *concentration* guard
  (`max_step_load`), so uncertain, non-beneficial and budget-dominating cases
  stay on the cheaper model;
* `axk1-think` is only reachable in Premium, only from `ax31`, and only inside
  a separate bounded sub-budget.

The two per-episode guards answer different failure modes, both found by
``tools/stress_safe_margin.py``:

``max_step_ratio``
    bounds how much more the upgraded model is predicted to cost *for the same
    prompt*. It is denominated in the light model's own predicted cost, a
    published per-prompt quantity rather than a batch statistic, so a batch of
    uniformly expansion-heavy prompts fails toward fewer upgrades instead of
    silently keeping the same promotion rate.

``max_step_load``
    bounds the predicted increment as a multiple of the batch's *mean* light
    cost, so one long prompt can never own a large share of a tier budget. It
    is scale-free in the batch size and therefore behaves the same on an
    880-episode and a 1,760-episode batch.

Neither guard can see the real driver of the cost tail, which is how many
output tokens the heavier model happens to emit. ``baselines/README.md``
records the measured limits of that blind spot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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
    load_policy,
    parse_submission,
    policy_sha256,
    submission_to_dict,
)

_BASELINES_DIRECTORY = str(Path(__file__).resolve().parent)
if _BASELINES_DIRECTORY not in sys.path:
    # Allow `python3 baselines/safe_margin.py` and importlib-based test loading
    # to reach the sibling baseline module the same way.
    sys.path.insert(0, _BASELINES_DIRECTORY)

from hash_regex import (  # noqa: E402  (sibling baseline module)
    DENSE_FEATURE_NAMES,
    HashRegexArtifact,
    LinearHead,
    load_artifact,
    raw_feature_vector,
)


STRATEGY_ID = "safe-margin"

#: Quantization used to build the content-derived allocation groups. Each entry
#: multiplies the matching :data:`DENSE_FEATURE_NAMES` value before truncation.
GROUP_QUANTIZATION: Tuple[float, ...] = (
    1.0,  # log_character_count
    1.0,  # log_word_count
    1.0,  # log_sentence_count
    2.0,  # log_message_count
    3.0,  # hangul_ratio
    1.0,  # log_code_marker_count
    1.0,  # log_math_marker_count
    8.0,  # numeric_density
    1.0,  # long_context
    1.0,  # log_reasoning_marker_count
    1.0,  # formal_reasoning
    1.0,  # program_analysis
    1.0,  # log_multi_constraint_count
    1.0,  # simple_transform
)

#: Number of efficiency buckets per doubling of predicted gain-per-cost. One
#: bucket per octave keeps groups large enough that a single optimistic
#: per-episode cost estimate cannot pull the whole group into the budget.
EFFICIENCY_BUCKETS_PER_OCTAVE = 1
#: Smallest efficiency still distinguished from "no measurable benefit".
MIN_EFFICIENCY = 1e-9

#: Extra conservatism applied on top of the floored per-model cost estimate.
COST_INFLATION: Mapping[str, float] = {
    "ax31-light": 1.0,
    "ax31": 1.10,
    "axk1-think": 1.35,
}


@dataclass(frozen=True)
class TierPlanConfig:
    """Frozen safety envelope for one evaluation tier.

    ``target_ratio`` is the planned *predicted* cost ratio. Because the cost
    estimate is deliberately floored and inflated, the realized ratio lands
    well below it; the public Dev figures in ``baselines/README.md`` record the
    calibration measured on public Train and verified on public Dev.

    ``*_max_step_ratio`` is the largest predicted cost of the upgraded model
    relative to the light model *on the same prompt*; ``*_max_step_load`` is
    the largest predicted increment relative to the batch's mean light cost.
    Both are calibrated from the deterministic resampling evidence in
    ``tools/stress_safe_margin.py``, not from whole-split averages alone.

    ``think_budget_share`` caps the fraction of the discretionary budget that
    the `axk1-think` stage may consume. It is a tail guard rather than a
    reservation: the cheap and reliable `ax31` upgrades always run first.
    """

    target_ratio: float
    ax31_min_gain: float
    ax31_max_step_ratio: float
    ax31_max_step_load: float
    allow_think: bool
    think_min_gain: float
    think_max_step_ratio: float
    think_max_step_load: float
    think_budget_share: float


#: Safety targets are expressed against the all-light baseline cost and stay
#: well below the public caps (1.25 / 2.0 / 4.0).
#:
#: Fast carries the tightest guards because it has the least absolute headroom:
#: on an 880-episode batch a single upgraded episode that happens to emit ~55x
#: its light generation moves the Fast ratio by roughly 0.06, which is most of
#: the distance between the realized ratio and the 1.25 cap. Balanced and
#: Premium can absorb the same episode, so they trade a looser guard for
#: quality. See the resampling table in ``baselines/README.md``.
TIER_PLAN_CONFIGS: Mapping[str, TierPlanConfig] = {
    "fast": TierPlanConfig(
        target_ratio=1.14,
        ax31_min_gain=0.008,
        ax31_max_step_ratio=3.0,
        ax31_max_step_load=4.0,
        allow_think=False,
        think_min_gain=1.0,
        think_max_step_ratio=0.0,
        think_max_step_load=0.0,
        think_budget_share=0.0,
    ),
    "balanced": TierPlanConfig(
        target_ratio=1.60,
        ax31_min_gain=0.004,
        ax31_max_step_ratio=5.0,
        ax31_max_step_load=6.0,
        allow_think=False,
        think_min_gain=1.0,
        think_max_step_ratio=0.0,
        think_max_step_load=0.0,
        think_budget_share=0.0,
    ),
    "premium": TierPlanConfig(
        target_ratio=3.20,
        ax31_min_gain=0.002,
        ax31_max_step_ratio=40.0,
        ax31_max_step_load=12.0,
        allow_think=True,
        think_min_gain=0.020,
        think_max_step_ratio=60.0,
        think_max_step_load=20.0,
        think_budget_share=0.65,
    ),
}


@dataclass(frozen=True)
class EpisodePrediction:
    """Per-episode prediction plus the content signature used for grouping."""

    scores: Mapping[str, float]
    costs: Mapping[str, float]
    signature: Tuple[int, ...]


@dataclass(frozen=True)
class StageReport:
    """Audit record for one upgrade ladder stage."""

    step: str
    considered: int
    eligible: int
    promoted: int
    groups_considered: int
    groups_promoted: int
    budget: float
    spent_ratio: float


@dataclass(frozen=True)
class SafeMarginPlan:
    submission: Submission
    predicted_budget_ratio: float
    target_ratio: float
    model_counts: Mapping[str, int]
    stages: Tuple[StageReport, ...]


def _rate_ratio(policy: RoutingPolicy, model_id: str) -> float:
    """Return the public input-token rate of ``model_id`` relative to light."""

    light = float(policy.models[policy.light_model_id].input_token_rate)
    if light <= 0:
        raise ProtocolError("light 모델의 input_token_rate는 0보다 커야 합니다.")
    return float(policy.models[model_id].input_token_rate) / light


def conservative_costs(
    raw_costs: Mapping[str, float], policy: RoutingPolicy
) -> Dict[str, float]:
    """Floor learned costs by the public rate ratio, then inflate them.

    The learned log-cost head is the only part of the public artifact that can
    drift badly on an unseen prompt mix. Flooring the estimate with a quantity
    that is fixed by the published cost policy keeps the router from ever
    treating an upgrade as cheaper than its input-token rate implies.
    """

    light_id = policy.light_model_id
    light = raw_costs[light_id]
    if not math.isfinite(light) or light <= 0:
        raise ValueError("light 모델의 예측 비용은 0보다 큰 유한값이어야 합니다.")
    result: Dict[str, float] = {}
    previous = 0.0
    for model_id in MODEL_IDS:
        floored = max(raw_costs[model_id], light * _rate_ratio(policy, model_id))
        value = floored * COST_INFLATION[model_id]
        # The ladder must stay monotone so an "upgrade" never looks cheaper.
        value = max(value, previous)
        result[model_id] = value
        previous = value
    return result


def content_signature(raw_features: Sequence[float]) -> Tuple[int, ...]:
    """Quantize the dense prompt features into a coarse content signature."""

    dense = raw_features[: len(DENSE_FEATURE_NAMES)]
    if len(dense) != len(DENSE_FEATURE_NAMES):
        raise ValueError("dense feature 길이가 현재 런타임과 다릅니다.")
    return tuple(
        int(math.floor(value * quantum))
        for value, quantum in zip(dense, GROUP_QUANTIZATION)
    )


def _efficiency_bucket(efficiency: float) -> int:
    """Return a deterministic log-spaced bucket for a gain-per-cost value."""

    if not math.isfinite(efficiency) or efficiency <= MIN_EFFICIENCY:
        return -(1 << 30)
    return int(
        math.floor(math.log2(efficiency) * EFFICIENCY_BUCKETS_PER_OCTAVE)
    )


def _linear(head: LinearHead, values: Sequence[float]) -> float:
    return head.intercept + math.fsum(
        coefficient * value
        for coefficient, value in zip(head.coefficients, values)
    )


def predict_from_raw(
    raw_features: Sequence[float], artifact: HashRegexArtifact
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Score one already-extracted feature vector.

    This mirrors :func:`hash_regex.predict_episode` numerically but takes the
    vector the caller already built for the content signature instead of
    extracting it a second time, which halves the per-episode feature work
    inside the runtime budget. ``tests/test_safe_margin_router.py`` pins the
    equivalence.
    """

    standardized = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            raw_features, artifact.feature_mean, artifact.feature_scale
        )
    )
    scores = {
        model_id: min(
            1.0, max(0.0, _linear(artifact.score_heads[model_id], standardized))
        )
        for model_id in MODEL_IDS
    }
    costs = {
        model_id: math.exp(
            min(
                50.0,
                max(-50.0, _linear(artifact.log_cost_heads[model_id], standardized)),
            )
        )
        for model_id in MODEL_IDS
    }
    light = costs[MODEL_IDS[0]]
    costs[MODEL_IDS[1]] = max(costs[MODEL_IDS[1]], light * (1.0 + 1e-12))
    costs[MODEL_IDS[2]] = max(
        costs[MODEL_IDS[2]], costs[MODEL_IDS[1]] * (1.0 + 1e-12)
    )
    return scores, costs


def predict_batch(
    episodes: Sequence[Episode],
    artifact: HashRegexArtifact,
    policy: RoutingPolicy,
) -> Tuple[EpisodePrediction, ...]:
    """Predict quality and conservative cost for every episode in the batch."""

    predictions: List[EpisodePrediction] = []
    for episode in episodes:
        raw = raw_feature_vector(episode, artifact.hash_bins)
        scores, raw_costs = predict_from_raw(raw, artifact)
        predictions.append(
            EpisodePrediction(
                scores=scores,
                costs=conservative_costs(raw_costs, policy),
                signature=content_signature(raw),
            )
        )
    return tuple(predictions)


def _run_stage(
    *,
    step_from: str,
    step_to: str,
    selected: List[str],
    predictions: Sequence[EpisodePrediction],
    light_total: float,
    mean_light: float,
    current_total: float,
    budget: float,
    min_gain: float,
    max_step_ratio: float,
    max_step_load: float,
) -> Tuple[float, StageReport]:
    """Promote whole content-derived groups while the budget allows it."""

    groups: Dict[Tuple[int, ...], List[int]] = {}
    considered = 0
    eligible = 0
    for index, prediction in enumerate(predictions):
        if selected[index] != step_from:
            continue
        considered += 1
        gain = prediction.scores[step_to] - prediction.scores[step_from]
        increment = prediction.costs[step_to] - prediction.costs[step_from]
        step_ratio = prediction.costs[step_to] / prediction.costs[
            MODEL_IDS[0]
        ]
        # How much of an average episode's light cost this single upgrade
        # would add. This is the concentration guard: it is scale-free in the
        # batch size, so one long prompt can never own a large share of the
        # tier budget no matter how the surrounding mix is composed.
        step_load = increment / mean_light
        if (
            gain < min_gain
            or increment <= 0
            or step_ratio > max_step_ratio
            or step_load > max_step_load
        ):
            # Uncertain, non-beneficial, or tail-cost cases stay cheaper.
            continue
        eligible += 1
        efficiency = gain / step_load
        key = (_efficiency_bucket(efficiency),) + prediction.signature
        groups.setdefault(key, []).append(index)

    promoted = 0
    groups_promoted = 0
    total = current_total
    # Descending efficiency, then ascending content signature: both derived
    # from prompt content only, so input order can never break a tie.
    for key in sorted(groups, key=lambda item: (-item[0], item[1:])):
        members = groups[key]
        delta = math.fsum(
            predictions[index].costs[step_to] - predictions[index].costs[step_from]
            for index in members
        )
        if total + delta > budget:
            continue
        total += delta
        groups_promoted += 1
        for index in members:
            selected[index] = step_to
            promoted += 1

    return total, StageReport(
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
    predictions: Sequence[EpisodePrediction],
    policy: RoutingPolicy,
    tier: str,
    config: Optional[TierPlanConfig] = None,
) -> Tuple[Tuple[str, ...], float, Tuple[StageReport, ...]]:
    """Return one model per episode plus the predicted cost ratio and audit."""

    if tier not in TIERS:
        raise ProtocolError(f"알 수 없는 tier: {tier}")
    if not predictions:
        raise ValueError("예측 배열은 비어 있을 수 없습니다.")
    plan_config = TIER_PLAN_CONFIGS[tier] if config is None else config
    light_id, ax31_id, think_id = MODEL_IDS

    light_total = math.fsum(item.costs[light_id] for item in predictions)
    if light_total <= 0:
        raise ValueError("예측 light 비용 합계는 0보다 커야 합니다.")
    mean_light = light_total / len(predictions)

    budget_multiplier = float(policy.tiers[tier].budget_multiplier)
    # Never plan above the published cap even if a target were misconfigured.
    target_ratio = min(plan_config.target_ratio, budget_multiplier)
    full_budget = light_total * target_ratio

    selected: List[str] = [light_id] * len(predictions)
    stages: List[StageReport] = []

    # The cheap, reliable ax31 ladder step always runs first and against the
    # whole budget: it buys far more predicted quality per credit than the
    # think model and its realized cost ratio is structurally bounded.
    total, ax31_stage = _run_stage(
        step_from=light_id,
        step_to=ax31_id,
        selected=selected,
        predictions=predictions,
        light_total=light_total,
        mean_light=mean_light,
        current_total=light_total,
        budget=full_budget,
        min_gain=plan_config.ax31_min_gain,
        max_step_ratio=plan_config.ax31_max_step_ratio,
        max_step_load=plan_config.ax31_max_step_load,
    )
    stages.append(ax31_stage)

    if plan_config.allow_think:
        # `axk1-think` has the heaviest and least predictable cost tail, so it
        # only ever spends a bounded share of the discretionary budget.
        think_allowance = (full_budget - light_total) * (
            plan_config.think_budget_share
        )
        total, think_stage = _run_stage(
            step_from=ax31_id,
            step_to=think_id,
            selected=selected,
            predictions=predictions,
            light_total=light_total,
            mean_light=mean_light,
            current_total=total,
            budget=min(full_budget, total + think_allowance),
            min_gain=plan_config.think_min_gain,
            max_step_ratio=plan_config.think_max_step_ratio,
            max_step_load=plan_config.think_max_step_load,
        )
        stages.append(think_stage)

    if total > full_budget:  # pragma: no cover - defensive, stages never exceed
        return (
            tuple(light_id for _item in predictions),
            1.0,
            tuple(stages),
        )
    return tuple(selected), total / light_total, tuple(stages)


def make_safe_margin_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: HashRegexArtifact,
    tier: str,
    config: Optional[TierPlanConfig] = None,
) -> SafeMarginPlan:
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
    selected, ratio, stages = plan_selection(predictions, policy, tier, config)
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
    plan_config = TIER_PLAN_CONFIGS[tier] if config is None else config
    return SafeMarginPlan(
        submission=parse_submission(submission_to_dict(submission)),
        predicted_budget_ratio=ratio,
        target_ratio=plan_config.target_ratio,
        model_counts=counts,
        stages=stages,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="safe-margin-router",
        description="여유 있는 예산 마진을 유지하는 보수적 prompt-only 라우터",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
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
        artifact = load_artifact(args.artifact)
        plan = make_safe_margin_submission(inputs, policy, artifact, args.tier)
        write_submission_atomic(args.output, plan.submission)
    except (OSError, ProtocolError, ValueError, json.JSONDecodeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    counts = ", ".join(
        f"{model_id}={plan.model_counts[model_id]}" for model_id in MODEL_IDS
    )
    print(
        "OK: "
        f"{args.tier} 제출 파일을 생성했습니다 "
        f"(예측 비용 비율 {plan.predicted_budget_ratio:.6f}, "
        f"안전 목표 {plan.target_ratio:.2f}, {counts})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
