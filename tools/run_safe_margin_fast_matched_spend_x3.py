#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Run the one frozen safe-margin Fast matched-spend x3 experiment."""

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

import risk_validation  # noqa: E402
import safe_margin  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    Decision,
    InputBatch,
    OutcomeBatch,
    RoutingPolicy,
    Submission,
    load_bundled_policy,
    load_input,
    load_outcomes,
)
from ossp_router.scoring import score_submissions  # noqa: E402
from stress_safe_margin import bootstrap_indices, outcome_tables  # noqa: E402


EXPERIMENT_ID = "safe-margin-fast-matched-spend-x3-v1"
REPORT_TYPE = "safe-margin-fast-matched-spend-x3-report-v1"
BASE_COMMIT = "3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9"
PROTOCOL_PATH = ROOT / "experiments/safe-margin-fast-matched-spend-x3/protocol.v1.json"
EXPECTED_PROTOCOL_SHA256 = "e88e3fd341b4a2f3d6f1b9366e38108d975ed1b7f6f058e365252d2fb96f99ae"
DEFAULT_REPORT = ROOT / "experiments/safe-margin-fast-matched-spend-x3/report.v1.json"
DEFAULT_TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"
DEFAULT_TRAIN_OUTCOMES = ROOT / "data/train/outcomes.json"
DEFAULT_DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"
DEFAULT_DEV_OUTCOMES = ROOT / "data/dev/outcomes.json"
BUCKETS_PER_OCTAVE = 3
DEV_COMPARATOR = Decimal("0.673182")
REPORT_DIGITS = 12
LIGHT, AX31, THINK = MODEL_IDS


class ExperimentError(RuntimeError):
    """Frozen evidence is missing, malformed, inconsistent, or uncertain."""


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


def verify_protocol(path: Path = PROTOCOL_PATH) -> Mapping[str, Any]:
    if file_sha256(path) != EXPECTED_PROTOCOL_SHA256:
        raise ExperimentError("frozen protocol SHA-256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    try:
        valid = (
            value["base"]["commit"] == BASE_COMMIT
            and value["candidate"]["efficiency_buckets_per_octave"] == 3
            and value["candidate"]["scope"] == "fast light->ax31 only"
            and value["candidate"]["tuning_sweep"] is False
            and value["default_runtime"]["changed"] is False
            and value["default_runtime"]["efficiency_buckets_per_octave"]
            == safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE == 1
            and tuple(value["train_gate"]["benchmark_families"])
            == risk_validation.FAMILY_LABELS
            and value["safety_gate"]["resamples"]
            == risk_validation.DEFAULT_RESAMPLES
            and value["safety_gate"]["seed"] == risk_validation.DEFAULT_SEED
            and value["safety_gate"]["tolerance"] == 0.0
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ExperimentError("frozen protocol contents mismatch")
    return value


def _decimal(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise InvalidOperation
    result = Decimal(str(value))
    if not result.is_finite():
        raise InvalidOperation
    return result


def _bucket(efficiency: float, buckets: int) -> int:
    if buckets <= 0:
        raise ValueError("bucket count must be positive")
    if not math.isfinite(efficiency) or efficiency <= safe_margin.MIN_EFFICIENCY:
        return -(1 << 30)
    return int(math.floor(math.log2(efficiency) * buckets))


def _eligible_groups(
    predictions: Sequence[safe_margin.EpisodePrediction],
    policy: RoutingPolicy,
) -> Mapping[Tuple[int, ...], Tuple[int, ...]]:
    """Return x3 Fast groups under the unmodified x1 eligibility guards."""

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
        values = (gain, increment, step_ratio, step_load)
        if not all(math.isfinite(value) for value in values):
            raise ExperimentError("non-finite prediction evidence")
        if (
            gain < config.ax31_min_gain
            or increment <= 0
            or step_ratio > config.ax31_max_step_ratio
            or step_load > config.ax31_max_step_load
        ):
            continue
        key = (_bucket(gain / step_load, BUCKETS_PER_OCTAVE),) + prediction.signature
        groups.setdefault(key, []).append(index)
    return {key: tuple(indices) for key, indices in groups.items()}


def matched_fast_plan(
    predictions: Sequence[safe_margin.EpisodePrediction], policy: RoutingPolicy
) -> CandidatePlan:
    """Start at x1 and apply only exact, cheaper x3-rank group swaps.

    Incoming groups are visited from best x3 rank to worst. For each incoming
    group, outgoing groups of the same cardinality are visited from worst x3
    rank to best. A swap is accepted only when the incoming group is strictly
    x3-preferred and its predicted incremental step load is no greater. Ties
    are resolved by the full content-derived group key. If no exact safe mate
    exists, the x1 decision is retained.
    """

    baseline, baseline_ratio, _stages = safe_margin.plan_selection(
        predictions, policy, "fast"
    )
    selected = list(baseline)
    groups = _eligible_groups(predictions, policy)
    baseline_selected = {
        key for key, members in groups.items() if all(baseline[i] == AX31 for i in members)
    }
    # A partly selected x3 group would be inconsistent with group atomicity.
    if any(
        any(baseline[i] == AX31 for i in members)
        and not all(baseline[i] == AX31 for i in members)
        for members in groups.values()
    ):
        raise ExperimentError("x1 selection splits an x3 content group")

    def increment(key: Tuple[int, ...]) -> float:
        return math.fsum(
            predictions[i].costs[AX31] - predictions[i].costs[LIGHT]
            for i in groups[key]
        )

    selected_groups = set(baseline_selected)
    swaps = 0
    incoming_order = sorted(
        (key for key in groups if key not in baseline_selected),
        key=lambda key: (-key[0], key[1:]),
    )
    for incoming in incoming_order:
        outgoing_order = sorted(
            (
                key
                for key in selected_groups & baseline_selected
                if len(groups[key]) == len(groups[incoming])
                and (-incoming[0], incoming[1:]) < (-key[0], key[1:])
                and increment(incoming) <= increment(key) + 1e-12
            ),
            key=lambda key: (key[0], tuple(-value for value in key[1:])),
        )
        if not outgoing_order:
            continue
        outgoing = outgoing_order[0]
        for index in groups[outgoing]:
            selected[index] = LIGHT
        for index in groups[incoming]:
            selected[index] = AX31
        selected_groups.remove(outgoing)
        selected_groups.add(incoming)
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
    promoted = selected.count(AX31)
    if (
        promoted != baseline.count(AX31)
        or selected.count(THINK) != 0
        or candidate_increment > baseline_increment + 1e-12
    ):
        raise ExperimentError("matched-swap invariant failed")
    return CandidatePlan(
        tuple(selected),
        (light_total + candidate_increment) / light_total,
        baseline_ratio,
        promoted,
        swaps,
        candidate_increment,
        baseline_increment,
    )


def candidate_submission(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: Any,
    tier: str,
    predictions: Optional[Sequence[safe_margin.EpisodePrediction]] = None,
) -> Tuple[Submission, CandidatePlan | None]:
    baseline = safe_margin.make_safe_margin_submission(inputs, policy, artifact, tier)
    if tier != "fast":
        return baseline.submission, None
    prediction_values = (
        tuple(predictions)
        if predictions is not None
        else safe_margin.predict_batch(inputs.episodes, artifact, policy)
    )
    plan = matched_fast_plan(prediction_values, policy)
    submission = Submission(
        schema_version=inputs.schema_version,
        challenge_id=inputs.challenge_id,
        policy_id=policy.policy_id,
        split=inputs.split,
        tier=tier,
        decisions=tuple(
            Decision(episode.episode_id, model)
            for episode, model in zip(inputs.episodes, plan.selected)
        ),
    )
    return submission, plan


def decide_train_gate(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    names = risk_validation.FAMILY_LABELS
    checks = {
        "weighted_score_delta_positive": False,
        "fast_quality_delta_positive": False,
        "positive_families": False,
        "nonnegative_families": False,
        "worst_family_delta": False,
        "balanced_premium_exact_x1": False,
        "fast_model_counts_exact_x1": False,
        "candidate_fast_predicted_spend_lte_x1": False,
        "deterministic_repeated_output": False,
    }
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
            "checks": checks,
            "gate_passed": all(checks.values()),
            "malformed": False,
            "positive_families": positive,
            "nonnegative_families": nonnegative,
            "worst_family_delta": str(worst),
        }
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return {"checks": checks, "gate_passed": False, "malformed": True}


def decide_dev_gate(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = {
        "score_strictly_above_threshold": False,
        "balanced_premium_exact_x1": False,
        "fast_model_counts_exact_x1": False,
        "candidate_fast_predicted_spend_lte_x1": False,
    }
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


def decide_safety_gate(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    checks: Dict[str, bool] = {}
    malformed = False
    try:
        for tier in TIERS:
            ref = reference["tiers"][tier]
            measured = candidate["tiers"][tier]
            checks[f"{tier}.bootstrap_cap_breaches"] = int(measured["bootstrap"]["cap_breaches"]) == 0
            checks[f"{tier}.whole_split_budget"] = measured["whole_split"]["budget_passed"] is True
            for family in risk_validation.FAMILY_LABELS:
                holdout = measured["family_holdouts"][family]
                checks[f"{tier}.holdout.{family}.budget"] = holdout["evaluated"] is True and holdout["budget_passed"] is True
            for statistic in ("p99", "max"):
                checks[f"{tier}.{statistic}_no_worse"] = _decimal(measured["bootstrap"]["cost_ratio"][statistic]) <= _decimal(ref["bootstrap"]["cost_ratio"][statistic])
    except (InvalidOperation, KeyError, TypeError, ValueError, OverflowError):
        malformed = True
    return {"checks": checks, "gate_passed": bool(checks) and not malformed and all(checks.values()), "malformed": malformed}


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
    return EvaluationData(inputs, outcomes, policy, costs, scores, families, input_path, outcomes_path)


def _plans(data: EvaluationData) -> Tuple[Tuple[Submission, ...], Tuple[Submission, ...], CandidatePlan]:
    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    predictions = safe_margin.predict_batch(data.inputs.episodes, artifact, data.policy)
    baseline = tuple(safe_margin.make_safe_margin_submission(data.inputs, data.policy, artifact, tier).submission for tier in TIERS)
    pairs = tuple(candidate_submission(data.inputs, data.policy, artifact, tier, predictions) for tier in TIERS)
    fast_plan = pairs[0][1]
    if fast_plan is None:
        raise ExperimentError("missing Fast candidate plan")
    return baseline, tuple(pair[0] for pair in pairs), fast_plan


def _models(submission: Submission) -> Tuple[str, ...]:
    return tuple(decision.model_id for decision in submission.decisions)


def _quality(data: EvaluationData, submission: Submission) -> Decimal:
    chosen = {decision.episode_id: decision.model_id for decision in submission.decisions}
    lookup = {(row.episode_id, row.model_id): row.score for row in data.outcomes.outcomes}
    with localcontext() as context:
        context.prec = 80
        return sum((lookup[(episode.episode_id, chosen[episode.episode_id])] for episode in data.inputs.episodes), Decimal(0)) / Decimal(len(data.inputs.episodes))


def _family_deltas(data: EvaluationData, baseline: Sequence[Submission], candidate: Sequence[Submission]) -> Mapping[str, str]:
    lookup = {(row.episode_id, row.model_id): row.score for row in data.outcomes.outcomes}
    base_models = {sub.tier: {d.episode_id: d.model_id for d in sub.decisions} for sub in baseline}
    candidate_models = {sub.tier: {d.episode_id: d.model_id for d in sub.decisions} for sub in candidate}
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
                for i in indices:
                    episode_id = data.inputs.episodes[i].episode_id
                    total += lookup[(episode_id, candidate_models[tier][episode_id])] - lookup[(episode_id, base_models[tier][episode_id])]
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
        "weighted_score_delta": str(_decimal(candidate_score["final_score"]) - _decimal(baseline_score["final_score"])),
        "fast_quality_delta": str(_quality(data, candidate[0]) - _quality(data, baseline[0])),
        "balanced_premium_exact_x1": all(_models(candidate[i]) == _models(baseline[i]) for i in (1, 2)),
        "fast_model_counts_exact_x1": sorted(_models(candidate[0])) == sorted(_models(baseline[0])),
        "candidate_fast_predicted_spend_lte_x1": fast_plan.predicted_increment <= fast_plan.baseline_increment + 1e-12,
        "deterministic_repeated_output": repeated,
        "fast_plan": {
            "baseline_predicted_ratio": fast_plan.baseline_ratio,
            "candidate_predicted_ratio": fast_plan.predicted_ratio,
            "baseline_increment": fast_plan.baseline_increment,
            "candidate_increment": fast_plan.predicted_increment,
            "promoted_episodes": fast_plan.promoted,
            "matched_swaps": fast_plan.swaps,
        },
    }


def evaluate_train(train_input: Path = DEFAULT_TRAIN_INPUT, train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES) -> Mapping[str, Any]:
    data = _load_split_once("train", train_input, train_outcomes)
    evidence = _common_evidence(data)
    family_deltas = _family_deltas(data, evidence["baseline"], evidence["candidate"])
    gate = decide_train_gate({**evidence, "family_deltas": family_deltas})
    return {
        "baseline_score": evidence["baseline_score"],
        "candidate_score": evidence["candidate_score_detail"],
        "weighted_score_delta": evidence["weighted_score_delta"],
        "fast_quality_delta": evidence["fast_quality_delta"],
        "family_deltas": family_deltas,
        "family_counts": {name: data.families.count(name) for name in risk_validation.FAMILY_LABELS},
        "balanced_premium_exact_x1": evidence["balanced_premium_exact_x1"],
        "fast_model_counts_exact_x1": evidence["fast_model_counts_exact_x1"],
        "candidate_fast_predicted_spend_lte_x1": evidence["candidate_fast_predicted_spend_lte_x1"],
        "deterministic_repeated_output": evidence["deterministic_repeated_output"],
        "fast_plan": evidence["fast_plan"],
        "input_path": _repo_path(train_input), "input_sha256": file_sha256(train_input),
        "outcomes_path": _repo_path(train_outcomes), "outcomes_sha256": file_sha256(train_outcomes),
        **gate,
    }


def evaluate_dev_loaded(data: EvaluationData) -> Mapping[str, Any]:
    evidence = _common_evidence(data)
    gate = decide_dev_gate(evidence)
    return {
        "accessed": True, "evaluation_count": 1,
        "baseline_score": evidence["baseline_score"],
        "candidate_score": evidence["candidate_score_detail"],
        "candidate_score_threshold_exclusive": str(DEV_COMPARATOR),
        "balanced_premium_exact_x1": evidence["balanced_premium_exact_x1"],
        "fast_model_counts_exact_x1": evidence["fast_model_counts_exact_x1"],
        "candidate_fast_predicted_spend_lte_x1": evidence["candidate_fast_predicted_spend_lte_x1"],
        "fast_plan": evidence["fast_plan"],
        "input_path": _repo_path(data.input_path), "input_sha256": file_sha256(data.input_path),
        "outcomes_path": _repo_path(data.outcomes_path), "outcomes_sha256": file_sha256(data.outcomes_path),
        **gate,
    }


def _risk_data(data: EvaluationData) -> risk_validation.SplitData:
    return risk_validation.SplitData(split=data.inputs.split, input_path=data.input_path, outcomes_path=data.outcomes_path, policy=data.policy, inputs=data.inputs, costs=data.costs, scores=data.scores, families=data.families)


def _candidate_risk_runner(data: risk_validation.SplitData) -> risk_validation.PolicyRunner:
    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    predictions = safe_margin.predict_batch(data.inputs.episodes, artifact, data.policy)
    def plan(indices: Sequence[int], tier: str) -> Tuple[str, ...]:
        subset = tuple(predictions[index] for index in indices)
        if tier == "fast":
            return matched_fast_plan(subset, data.policy).selected
        return safe_margin.plan_selection(subset, data.policy, tier)[0]
    return risk_validation.PolicyRunner(name=EXPERIMENT_ID, artifact_path=safe_margin.DEFAULT_ARTIFACT_PATH, artifact_sha256=file_sha256(safe_margin.DEFAULT_ARTIFACT_PATH), plan=plan)


def evaluate_safety_loaded(data: EvaluationData) -> Mapping[str, Any]:
    measured = _risk_data(data)
    indices = bootstrap_indices(measured.num_episodes, risk_validation.DEFAULT_RESAMPLES, risk_validation.DEFAULT_SEED)
    reference = risk_validation.evaluate_runner(measured, risk_validation.safe_margin_runner(measured), indices)
    candidate = risk_validation.evaluate_runner(measured, _candidate_risk_runner(measured), indices)
    return {"accessed": True, "evaluation_count": 1, "resamples": risk_validation.DEFAULT_RESAMPLES, "seed": risk_validation.DEFAULT_SEED, "reference": reference, "candidate": candidate, **decide_safety_gate(reference, candidate)}


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


def run_experiment(*, train_input: Path = DEFAULT_TRAIN_INPUT, train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES, dev_input: Path = DEFAULT_DEV_INPUT, dev_outcomes: Path = DEFAULT_DEV_OUTCOMES, report_path: Path = DEFAULT_REPORT) -> Mapping[str, Any]:
    protocol = verify_protocol()
    report: Dict[str, Any] = {
        "base_commit": BASE_COMMIT, "experiment_id": EXPERIMENT_ID, "report_type": REPORT_TYPE,
        "protocol_path": _repo_path(PROTOCOL_PATH), "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "protocol_version": protocol["version"],
        "candidate": {"efficiency_buckets_per_octave": 3, "scope": "fast light->ax31 only", "tuning_sweep": False},
        "production_default": {"changed": False, "efficiency_buckets_per_octave": safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE},
        "decision": {"eligible": False, "reason": "Train gate not evaluated", "submission_default": "safe-margin"},
        "dev": {"accessed": False, "evaluation_count": 0, "gate_passed": False},
        "safety": {"accessed": False, "evaluation_count": 0, "gate_passed": False, "resamples": 0},
    }
    report["train"] = evaluate_train(train_input, train_outcomes)
    if not report["train"]["gate_passed"]:
        report["decision"]["reason"] = "Train gate failed; Dev and safety remained closed"
    else:
        dev_data = _load_split_once("dev", dev_input, dev_outcomes)
        report["dev"] = evaluate_dev_loaded(dev_data)
        if not report["dev"]["gate_passed"]:
            report["decision"]["reason"] = "strict Public Dev gate failed; safety remained closed"
        else:
            report["safety"] = evaluate_safety_loaded(dev_data)
            if report["safety"]["gate_passed"]:
                report["decision"] = {"eligible": True, "reason": "all frozen gates passed; eligible for later owner consideration", "submission_default": "safe-margin"}
            else:
                report["decision"]["reason"] = "5,000-resample safety gate failed"
    rounded = _round(report)
    _atomic_json(report_path, rounded)
    return rounded


def reemit_existing_report(report_path: Path = DEFAULT_REPORT) -> Mapping[str, Any]:
    verify_protocol()
    try:
        value = json.loads(report_path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ExperimentError(f"cannot re-emit malformed terminal report: {exc}") from exc
    if not isinstance(value, dict) or value.get("report_type") != REPORT_TYPE or value.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256 or value.get("experiment_id") != EXPERIMENT_ID or value.get("decision", {}).get("submission_default") != "safe-margin":
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
        report = run_experiment(train_input=args.train_input, train_outcomes=args.train_outcomes, dev_input=args.dev_input, dev_outcomes=args.dev_outcomes, report_path=args.report)
    except (ExperimentError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Train={'PASS' if report['train']['gate_passed'] else 'FAIL'}; Dev evaluations={report['dev']['evaluation_count']}; safety evaluations={report['safety']['evaluation_count']}; eligible={report['decision']['eligible']}")
    return 0 if report["decision"]["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
