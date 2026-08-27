#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Run the single frozen safe-margin efficiency-bucket x3 experiment.

Public Dev is not opened until the complete Train gate passes. If opened, the
Dev input and outcomes are loaded exactly once and the same in-memory evidence
is reused by the conditional 5,000-resample safety gate.
"""

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
    TIERS,
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


EXPERIMENT_ID = "safe-margin-efficiency-bucket-x3-v1"
REPORT_TYPE = "safe-margin-efficiency-bucket-x3-report-v1"
BASE_COMMIT = "3fbdfe84f7a7ccee247d3cc0536d11ada6a21ec9"
PROTOCOL_PATH = ROOT / "experiments/safe-margin-efficiency-bucket-x3/protocol.v1.json"
EXPECTED_PROTOCOL_SHA256 = "40d7e144fe7acda8440d2515fd280887aa6aa5ef4525826aba4f298a1ad4f683"
DEFAULT_REPORT = ROOT / "experiments/safe-margin-efficiency-bucket-x3/report.v1.json"
DEFAULT_TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"
DEFAULT_TRAIN_OUTCOMES = ROOT / "data/train/outcomes.json"
DEFAULT_DEV_INPUT = ROOT / "data/materialized/dev/inputs.json"
DEFAULT_DEV_OUTCOMES = ROOT / "data/dev/outcomes.json"
CANDIDATE_BUCKETS_PER_OCTAVE = 3
TRAIN_MIN_POSITIVE_FAMILIES = 4
TRAIN_MIN_NONNEGATIVE_FAMILIES = 9
TRAIN_WORST_FAMILY_DELTA = Decimal("-0.0005")
DEV_COMPARATOR = Decimal("0.673182")
REPORT_DIGITS = 12


class ExperimentError(RuntimeError):
    """The frozen experiment cannot produce certain, comparable evidence."""


@dataclass(frozen=True)
class EvaluationData:
    """One split loaded once with its realized outcome lookup tables."""

    inputs: InputBatch
    outcomes: OutcomeBatch
    policy: RoutingPolicy
    costs: Tuple[Mapping[str, float], ...]
    scores: Tuple[Mapping[str, float], ...]
    families: Tuple[str, ...]
    input_path: Path
    outcomes_path: Path


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
            and value["candidate"]["efficiency_buckets_per_octave"]
            == CANDIDATE_BUCKETS_PER_OCTAVE
            and value["candidate"]["other_bucket_counts_evaluated"] == 0
            and value["default_runtime"]["efficiency_buckets_per_octave"]
            == safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE
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


def decide_train_gate(
    weighted_score_delta: Any, family_deltas: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Apply every exact Train boundary; malformed evidence fails closed."""

    checks = {
        "weighted_score_delta_positive": False,
        "positive_families": False,
        "nonnegative_families": False,
        "worst_family_delta": False,
    }
    try:
        if set(family_deltas) != set(risk_validation.FAMILY_LABELS):
            raise InvalidOperation
        delta = _decimal(weighted_score_delta)
        values = [_decimal(family_deltas[name]) for name in risk_validation.FAMILY_LABELS]
        positive = sum(value > 0 for value in values)
        nonnegative = sum(value >= 0 for value in values)
        worst = min(values)
        checks = {
            "weighted_score_delta_positive": delta > 0,
            "positive_families": positive >= TRAIN_MIN_POSITIVE_FAMILIES,
            "nonnegative_families": nonnegative >= TRAIN_MIN_NONNEGATIVE_FAMILIES,
            "worst_family_delta": worst >= TRAIN_WORST_FAMILY_DELTA,
        }
        return {
            "checks": checks,
            "gate_passed": all(checks.values()),
            "malformed": False,
            "nonnegative_families": nonnegative,
            "positive_families": positive,
            "worst_family_delta": str(worst),
        }
    except (InvalidOperation, TypeError, ValueError):
        return {
            "checks": checks,
            "gate_passed": False,
            "malformed": True,
            "nonnegative_families": 0,
            "positive_families": 0,
            "worst_family_delta": None,
        }


def decide_dev_gate(candidate_score: Any) -> Mapping[str, Any]:
    """The frozen Public Dev comparison is strict and fails closed."""

    try:
        score = _decimal(candidate_score)
        passed = score > DEV_COMPARATOR
        return {"gate_passed": passed, "malformed": False}
    except (InvalidOperation, TypeError, ValueError):
        return {"gate_passed": False, "malformed": True}


def decide_safety_gate(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Require exact p99/max preservation and zero cap/budget breaches."""

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
                    holdout["evaluated"] is True
                    and holdout["budget_passed"] is True
                )
            for statistic in ("p99", "max"):
                ref_value = _decimal(ref["bootstrap"]["cost_ratio"][statistic])
                candidate_value = _decimal(
                    measured["bootstrap"]["cost_ratio"][statistic]
                )
                checks[f"{tier}.{statistic}_no_worse"] = candidate_value <= ref_value
    except (InvalidOperation, KeyError, TypeError, ValueError, OverflowError):
        malformed = True
    return {
        "checks": checks,
        "gate_passed": bool(checks) and not malformed and all(checks.values()),
        "malformed": malformed,
    }


def _load_split_once(
    split: str, input_path: Path, outcomes_path: Path
) -> EvaluationData:
    inputs = load_input(input_path)
    outcomes = load_outcomes(outcomes_path)
    if inputs.split != split or outcomes.split != split:
        raise ExperimentError(f"expected the public {split} split")
    policy = load_bundled_policy()
    costs, scores = outcome_tables(inputs, outcomes, policy)
    families = risk_validation.reconstruct_families(split, inputs)
    if set(families) != set(risk_validation.FAMILY_LABELS):
        raise ExperimentError(f"{split} does not contain all nine frozen families")
    return EvaluationData(
        inputs, outcomes, policy, costs, scores, families, input_path, outcomes_path
    )


def _submissions(
    inputs: InputBatch,
    policy: RoutingPolicy,
    artifact: Any,
    buckets_per_octave: int,
) -> Tuple[Submission, ...]:
    return tuple(
        safe_margin.make_safe_margin_submission(
            inputs,
            policy,
            artifact,
            tier,
            efficiency_buckets_per_octave=buckets_per_octave,
        ).submission
        for tier in TIERS
    )


def _family_deltas(
    data: EvaluationData,
    baseline: Sequence[Submission],
    candidate: Sequence[Submission],
) -> Mapping[str, str]:
    outcome_index = {
        (item.episode_id, item.model_id): item for item in data.outcomes.outcomes
    }
    base_models = {
        submission.tier: {
            decision.episode_id: decision.model_id
            for decision in submission.decisions
        }
        for submission in baseline
    }
    candidate_models = {
        submission.tier: {
            decision.episode_id: decision.model_id
            for decision in submission.decisions
        }
        for submission in candidate
    }
    result: Dict[str, str] = {}
    with localcontext() as context:
        context.prec = 80
        for family in risk_validation.FAMILY_LABELS:
            indices = [
                index for index, label in enumerate(data.families) if label == family
            ]
            if not indices:
                raise ExperimentError(f"family {family} has no rows")
            delta = Decimal("0")
            for tier in TIERS:
                total = Decimal("0")
                for index in indices:
                    episode_id = data.inputs.episodes[index].episode_id
                    left = outcome_index[(episode_id, base_models[tier][episode_id])]
                    right = outcome_index[(episode_id, candidate_models[tier][episode_id])]
                    total += right.score - left.score
                delta += data.policy.tiers[tier].weight * total / Decimal(len(indices))
            result[family] = str(delta)
    return result


def evaluate_train(
    train_input: Path = DEFAULT_TRAIN_INPUT,
    train_outcomes: Path = DEFAULT_TRAIN_OUTCOMES,
) -> Mapping[str, Any]:
    data = _load_split_once("train", train_input, train_outcomes)
    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    baseline = _submissions(
        data.inputs, data.policy, artifact, safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE
    )
    candidate = _submissions(
        data.inputs, data.policy, artifact, CANDIDATE_BUCKETS_PER_OCTAVE
    )
    baseline_score = score_submissions(data.inputs, data.outcomes, baseline, data.policy)
    candidate_score = score_submissions(data.inputs, data.outcomes, candidate, data.policy)
    weighted_delta = _decimal(candidate_score["final_score"]) - _decimal(
        baseline_score["final_score"]
    )
    family_deltas = _family_deltas(data, baseline, candidate)
    gate = decide_train_gate(weighted_delta, family_deltas)
    repeated = candidate == _submissions(
        data.inputs, data.policy, artifact, CANDIDATE_BUCKETS_PER_OCTAVE
    )
    if not repeated:
        gate = dict(gate)
        gate["gate_passed"] = False
        gate["checks"] = {**gate["checks"], "deterministic_repeated_output": False}
    else:
        gate = dict(gate)
        gate["checks"] = {**gate["checks"], "deterministic_repeated_output": True}
    return {
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "family_counts": {
            family: data.families.count(family)
            for family in risk_validation.FAMILY_LABELS
        },
        "family_deltas": family_deltas,
        **gate,
        "input_path": _repo_path(train_input),
        "input_sha256": file_sha256(train_input),
        "outcomes_path": _repo_path(train_outcomes),
        "outcomes_sha256": file_sha256(train_outcomes),
        "weighted_score_delta": str(weighted_delta),
    }


def evaluate_dev_loaded(data: EvaluationData) -> Mapping[str, Any]:
    artifact = safe_margin.load_artifact(safe_margin.DEFAULT_ARTIFACT_PATH)
    baseline = _submissions(
        data.inputs, data.policy, artifact, safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE
    )
    candidate = _submissions(
        data.inputs, data.policy, artifact, CANDIDATE_BUCKETS_PER_OCTAVE
    )
    baseline_score = score_submissions(data.inputs, data.outcomes, baseline, data.policy)
    candidate_score = score_submissions(data.inputs, data.outcomes, candidate, data.policy)
    gate = decide_dev_gate(candidate_score.get("final_score"))
    return {
        "accessed": True,
        "evaluation_count": 1,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "candidate_score_threshold_exclusive": str(DEV_COMPARATOR),
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
    predictions = safe_margin.predict_batch(data.inputs.episodes, artifact, data.policy)

    def plan(indices: Sequence[int], tier: str) -> Tuple[str, ...]:
        selected, _ratio, _stages = safe_margin.plan_selection(
            [predictions[index] for index in indices],
            data.policy,
            tier,
            efficiency_buckets_per_octave=CANDIDATE_BUCKETS_PER_OCTAVE,
        )
        return selected

    return risk_validation.PolicyRunner(
        name=EXPERIMENT_ID,
        artifact_path=safe_margin.DEFAULT_ARTIFACT_PATH,
        artifact_sha256=file_sha256(safe_margin.DEFAULT_ARTIFACT_PATH),
        plan=plan,
    )


def evaluate_safety_loaded(data: EvaluationData) -> Mapping[str, Any]:
    measured_data = _risk_data(data)
    indices = bootstrap_indices(
        measured_data.num_episodes,
        risk_validation.DEFAULT_RESAMPLES,
        risk_validation.DEFAULT_SEED,
    )
    reference = risk_validation.evaluate_runner(
        measured_data, risk_validation.safe_margin_runner(measured_data), indices
    )
    candidate = risk_validation.evaluate_runner(
        measured_data, _candidate_risk_runner(measured_data), indices
    )
    gate = decide_safety_gate(reference, candidate)
    return {
        "accessed": True,
        "evaluation_count": 1,
        "resamples": risk_validation.DEFAULT_RESAMPLES,
        "seed": risk_validation.DEFAULT_SEED,
        "reference": reference,
        "candidate": candidate,
        **gate,
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
        "candidate": {"efficiency_buckets_per_octave": 3, "tuning_sweep": False},
        "decision": {
            "eligible": False,
            "reason": "Train gate not evaluated",
            "submission_default": "safe-margin",
        },
        "dev": {"accessed": False, "evaluation_count": 0, "gate_passed": False},
        "experiment_id": EXPERIMENT_ID,
        "production_default_efficiency_buckets_per_octave": (
            safe_margin.EFFICIENCY_BUCKETS_PER_OCTAVE
        ),
        "protocol_path": _repo_path(PROTOCOL_PATH),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_version": protocol["version"],
        "report_type": REPORT_TYPE,
        "safety": {
            "accessed": False,
            "evaluation_count": 0,
            "gate_passed": False,
            "resamples": 0,
        },
    }
    train = evaluate_train(train_input, train_outcomes)
    report["train"] = train
    if not train["gate_passed"]:
        report["decision"]["reason"] = "Train gate failed; Dev and safety remained closed"
    else:
        # This is the sole Dev load. The returned object is reused below.
        dev_data = _load_split_once("dev", dev_input, dev_outcomes)
        report["dev"] = evaluate_dev_loaded(dev_data)
        if not report["dev"]["gate_passed"]:
            report["decision"]["reason"] = "strict Public Dev score gate failed; safety remained closed"
        else:
            report["safety"] = evaluate_safety_loaded(dev_data)
            if report["safety"]["gate_passed"]:
                report["decision"] = {
                    "eligible": True,
                    "reason": "all frozen Train, Dev, and safety gates passed; eligible for later adoption",
                    "submission_default": "safe-margin",
                }
            else:
                report["decision"]["reason"] = "5,000-resample safety gate failed"
    rounded = _round(report)
    _atomic_json(report_path, rounded)
    return rounded


def reemit_existing_report(report_path: Path = DEFAULT_REPORT) -> Mapping[str, Any]:
    """Canonically regenerate a terminal report without reopening any data."""

    verify_protocol()
    try:
        value = json.loads(
            report_path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ExperimentError(f"cannot re-emit malformed terminal report: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("report_type") != REPORT_TYPE
        or value.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
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
    parser.add_argument(
        "--reemit-existing",
        action="store_true",
        help="canonically rewrite the terminal report without reading Train or Dev",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.reemit_existing:
            reemit_existing_report(args.report)
            print(f"Re-emitted byte-stable report: {_repo_path(args.report)}")
            return 0
        else:
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
