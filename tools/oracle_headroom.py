# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Development-only oracle headroom decomposition for the safe-margin router.

This tool is **not** part of the submitted container runtime and must never be
imported by it. It uses public outcome values *only* inside this development
analysis to answer one planning question before any new model is built: *how
much weighted score is reachable at all, and which prediction (quality or
cost) is the bottleneck?*

Per public split it reports five deterministic variants:

1. ``baseline``            – current quality predictions + current conservative
   cost predictions + current safe-margin planner;
2. ``oracle_gain``         – realized per-model quality substituted into the
   planner, current cost predictions kept;
3. ``oracle_cost``         – realized per-model costs substituted, current
   quality predictions kept;
4. ``oracle_both``         – realized quality and realized costs through the
   same planner and safety envelope;
5. ``safe_envelope_bound`` – an oracle *upper bound*: a knapsack selection on
   realized quality/cost whose realized budget is pushed only as far as the
   same-run 5,000-resample safe-margin reference allows (candidate p99 and
   maximum resample ratios may not exceed the reference, and no resample may
   breach the hard cap). Both the integral greedy solution and the fractional
   LP relaxation bound are reported.

The stop gate for building a larger learned router is
``safe_envelope_bound.lp_upper_bound.weighted >= 0.690000``.

```console
PYTHONPATH=src python3 tools/oracle_headroom.py --split dev
```
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "tools"))

from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    ProtocolError,
    write_json,
)

from run_mvp import DEFAULT_ARTIFACT, MvpRunError, load_safe_margin  # noqa: E402
from risk_validation import (  # noqa: E402  (sibling development tool)
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    RiskValidationError,
    SplitData,
    file_sha256,
    load_split_data,
    realized_ratio_quality,
    safe_margin_runner,
)
from stress_safe_margin import bootstrap_indices, quantile  # noqa: E402


REPORT_TYPE = "oracle-headroom-report"
STOP_GATE_TARGET = 0.690000
REPORT_DIGITS = 6
#: Budget binary-search precision, in realized cost-ratio units.
ENVELOPE_PRECISION = 5e-4

#: Final-score tier weights, fixed by the public policy (0.4 / 0.3 / 0.3).
TIER_WEIGHTS: Mapping[str, float] = {
    "fast": 0.4,
    "balanced": 0.3,
    "premium": 0.3,
}


def _require_numpy():
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RiskValidationError(
            "oracle headroom 분석에는 개발 전용 NumPy가 필요합니다."
        ) from exc
    return numpy


def weighted_score(per_tier: Mapping[str, float]) -> float:
    return math.fsum(
        TIER_WEIGHTS[tier] * per_tier[tier] for tier in TIERS
    )


def _predictions_with(
    predictions: Sequence[Any],
    data: SplitData,
    *,
    oracle_scores: bool,
    oracle_costs: bool,
):
    """Clone safe-margin predictions with oracle values substituted."""

    safe_margin = load_safe_margin()
    result = []
    for index, prediction in enumerate(predictions):
        scores = (
            dict(data.scores[index]) if oracle_scores else prediction.scores
        )
        costs = dict(data.costs[index]) if oracle_costs else prediction.costs
        result.append(
            safe_margin.EpisodePrediction(
                scores=scores,
                costs=costs,
                signature=prediction.signature,
            )
        )
    return tuple(result)


def _variant_metrics(
    data: SplitData, predictions: Sequence[Any]
) -> Dict[str, Any]:
    safe_margin = load_safe_margin()
    indices = tuple(range(data.num_episodes))
    result: Dict[str, Any] = {}
    per_tier_quality: Dict[str, float] = {}
    for tier in TIERS:
        selected, _ratio, _stages = safe_margin.plan_selection(
            predictions, data.policy, tier
        )
        ratio, quality = realized_ratio_quality(data, indices, selected)
        cap = float(data.policy.tiers[tier].budget_multiplier)
        per_tier_quality[tier] = quality
        result[tier] = {
            "cost_ratio": ratio,
            "quality_score": quality,
            "budget_passed": ratio <= cap,
            "model_counts": {
                model_id: sum(1 for item in selected if item == model_id)
                for model_id in MODEL_IDS
            },
        }
    result["weighted"] = weighted_score(per_tier_quality)
    return result


@dataclass(frozen=True)
class HullStep:
    """One incremental upgrade step on an episode's realized convex hull."""

    efficiency: float
    delta_cost: float
    delta_quality: float
    episode_index: int
    model_index: int


def oracle_base_and_steps(
    data: SplitData,
) -> Tuple[Tuple[int, ...], Tuple[HullStep, ...]]:
    """Return the free-quality base choice plus sorted hull upgrade steps.

    The base picks, per episode, the best realized score among models that
    cost no more than the light model (free upgrades are always taken). The
    remaining strictly-more-expensive, strictly-better points form each
    episode's convex hull; their incremental steps are returned sorted by
    descending realized efficiency.
    """

    base: List[int] = []
    steps: List[HullStep] = []
    for index in range(data.num_episodes):
        costs = [data.costs[index][model_id] for model_id in MODEL_IDS]
        scores = [data.scores[index][model_id] for model_id in MODEL_IDS]
        light_cost = costs[0]
        best = 0
        for j in range(1, len(MODEL_IDS)):
            if costs[j] <= light_cost and (
                scores[j] > scores[best]
                or (scores[j] == scores[best] and costs[j] < costs[best])
            ):
                best = j
        base.append(best)
        current_cost, current_score = costs[best], scores[best]
        remaining = [
            (costs[j], scores[j], j)
            for j in range(len(MODEL_IDS))
            if costs[j] > current_cost and scores[j] > current_score
        ]
        while remaining:
            candidate = max(
                remaining,
                key=lambda item: (
                    (item[1] - current_score) / (item[0] - current_cost),
                    -item[0],
                ),
            )
            steps.append(
                HullStep(
                    efficiency=(candidate[1] - current_score)
                    / (candidate[0] - current_cost),
                    delta_cost=candidate[0] - current_cost,
                    delta_quality=candidate[1] - current_score,
                    episode_index=index,
                    model_index=candidate[2],
                )
            )
            current_cost, current_score = candidate[0], candidate[1]
            remaining = [
                item
                for item in remaining
                if item[0] > current_cost and item[1] > current_score
            ]
    steps.sort(
        key=lambda step: (
            -step.efficiency,
            step.delta_cost,
            step.episode_index,
            step.model_index,
        )
    )
    return tuple(base), tuple(steps)


def greedy_selection(
    data: SplitData,
    base: Sequence[int],
    steps: Sequence[HullStep],
    target_ratio: float,
) -> Tuple[Tuple[int, ...], float]:
    """Fill the realized budget greedily; also return the LP fractional bonus."""

    light_total = math.fsum(
        data.costs[index][MODEL_IDS[0]] for index in range(data.num_episodes)
    )
    base_total = math.fsum(
        data.costs[index][MODEL_IDS[choice]]
        for index, choice in enumerate(base)
    )
    budget = target_ratio * light_total - base_total
    selection = list(base)
    spent = 0.0
    fractional_bonus = 0.0
    for step in steps:
        if spent + step.delta_cost <= budget:
            spent += step.delta_cost
            selection[step.episode_index] = step.model_index
        elif fractional_bonus == 0.0 and budget > spent:
            # First step that no longer fits: the LP relaxation takes the
            # remaining budget fractionally, which upper-bounds every integral
            # solution at this budget.
            fractional_bonus = (
                (budget - spent) / step.delta_cost * step.delta_quality
            )
    return tuple(selection), fractional_bonus


def _selection_quality(data: SplitData, selection: Sequence[int]) -> float:
    return (
        math.fsum(
            data.scores[index][MODEL_IDS[choice]]
            for index, choice in enumerate(selection)
        )
        / data.num_episodes
    )


def _selection_counts(selection: Sequence[int]) -> Dict[str, int]:
    return {
        model_id: sum(1 for choice in selection if choice == j)
        for j, model_id in enumerate(MODEL_IDS)
    }


class ResampleEvaluator:
    """Evaluate fixed per-episode selections over shared bootstrap resamples."""

    def __init__(self, data: SplitData, index_sets: Sequence[Sequence[int]]):
        numpy = _require_numpy()
        self._numpy = numpy
        self._data = data
        counts = numpy.zeros(
            (len(index_sets), data.num_episodes), dtype=numpy.float64
        )
        for row, indices in enumerate(index_sets):
            counts[row] += numpy.bincount(
                numpy.asarray(indices, dtype=numpy.int64),
                minlength=data.num_episodes,
            )
        self._counts = counts
        self._cost_matrix = numpy.asarray(
            [
                [data.costs[index][model_id] for model_id in MODEL_IDS]
                for index in range(data.num_episodes)
            ]
        )
        self._light_totals = counts @ self._cost_matrix[:, 0]

    def ratio_stats(self, selection: Sequence[int]) -> Dict[str, float]:
        numpy = self._numpy
        chosen = self._cost_matrix[
            numpy.arange(self._data.num_episodes),
            numpy.asarray(selection, dtype=numpy.int64),
        ]
        ratios = (self._counts @ chosen) / self._light_totals
        ordered = numpy.sort(ratios)
        values = ordered.tolist()
        return {
            "mean": float(numpy.mean(ordered)),
            "min": values[0],
            "max": values[-1],
            "p50": quantile(values, 0.5),
            "p95": quantile(values, 0.95),
            "p99": quantile(values, 0.99),
        }


def safe_envelope_bound(
    data: SplitData,
    reference_bootstrap: Mapping[str, Mapping[str, Any]],
    index_sets: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Largest oracle selection whose resample tail stays inside the reference.

    For each tier the realized budget ratio is binary-searched to the largest
    value whose fixed oracle selection satisfies, over the same resamples as
    the safe-margin reference: zero hard-cap breaches, p99 no worse than the
    reference p99, and maximum no worse than the reference maximum.
    """

    base, steps = oracle_base_and_steps(data)
    evaluator = ResampleEvaluator(data, index_sets)
    result: Dict[str, Any] = {}
    achievable: Dict[str, float] = {}
    lp_bound: Dict[str, float] = {}
    for tier in TIERS:
        cap = float(data.policy.tiers[tier].budget_multiplier)
        reference = reference_bootstrap[tier]["cost_ratio"]

        def feasible(target: float) -> Tuple[bool, Dict[str, float]]:
            selection, _bonus = greedy_selection(data, base, steps, target)
            stats = evaluator.ratio_stats(selection)
            ok = (
                stats["max"] <= cap
                and stats["p99"] <= reference["p99"]
                and stats["max"] <= reference["max"]
            )
            return ok, stats

        low, high = 1.0, cap
        ok_low, _stats = feasible(low)
        if not ok_low:  # pragma: no cover - all-light never breaches
            raise RiskValidationError(f"{tier}: 기준 예산에서 실패했습니다.")
        while high - low > ENVELOPE_PRECISION:
            middle = (low + high) / 2.0
            ok, _stats = feasible(middle)
            if ok:
                low = middle
            else:
                high = middle
        selection, bonus = greedy_selection(data, base, steps, low)
        stats = evaluator.ratio_stats(selection)
        indices = tuple(range(data.num_episodes))
        selected_models = tuple(
            MODEL_IDS[choice] for choice in selection
        )
        ratio, quality = realized_ratio_quality(
            data, indices, selected_models
        )
        achievable[tier] = quality
        lp_bound[tier] = quality + bonus / data.num_episodes
        result[tier] = {
            "budget_multiplier": cap,
            "envelope_target_ratio": low,
            "whole_split_cost_ratio": ratio,
            "achievable_quality": quality,
            "lp_upper_bound_quality": lp_bound[tier],
            "model_counts": _selection_counts(selection),
            "resample_cost_ratio": stats,
            "reference_p99": reference["p99"],
            "reference_max": reference["max"],
        }
    result["achievable_weighted"] = weighted_score(achievable)
    result["lp_upper_bound_weighted"] = weighted_score(lp_bound)
    return result


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, REPORT_DIGITS)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def _repo_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def analyze_split(
    *,
    split: str = "dev",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    artifact_path: Path = DEFAULT_ARTIFACT,
    input_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    policy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Produce the full oracle decomposition for one public split."""

    data = load_split_data(
        split,
        input_path=input_path,
        outcomes_path=outcomes_path,
        policy_path=policy_path,
    )
    safe_margin = load_safe_margin()
    artifact = safe_margin.load_artifact(artifact_path)
    predictions = safe_margin.predict_batch(
        data.inputs.episodes, artifact, data.policy
    )
    index_sets = bootstrap_indices(data.num_episodes, resamples, seed)

    # The 5,000-resample same-run reference: the current safe-margin planner
    # replanned per resample, exactly as the risk-validation gate measures it.
    runner = safe_margin_runner(data, artifact_path)
    reference_bootstrap = {}
    for tier in TIERS:
        cap = float(data.policy.tiers[tier].budget_multiplier)
        ratios: List[float] = []
        for indices in index_sets:
            selected = runner.plan(indices, tier)
            ratio, _quality = realized_ratio_quality(data, indices, selected)
            ratios.append(ratio)
        ordered = sorted(ratios)
        reference_bootstrap[tier] = {
            "cost_ratio": {
                "mean": math.fsum(ordered) / len(ordered),
                "min": ordered[0],
                "max": ordered[-1],
                "p50": quantile(ordered, 0.5),
                "p95": quantile(ordered, 0.95),
                "p99": quantile(ordered, 0.99),
            },
            "cap_breaches": sum(1 for value in ordered if value > cap),
        }

    variants = {
        "baseline": _variant_metrics(data, predictions),
        "oracle_gain": _variant_metrics(
            data,
            _predictions_with(
                predictions, data, oracle_scores=True, oracle_costs=False
            ),
        ),
        "oracle_cost": _variant_metrics(
            data,
            _predictions_with(
                predictions, data, oracle_scores=False, oracle_costs=True
            ),
        ),
        "oracle_both": _variant_metrics(
            data,
            _predictions_with(
                predictions, data, oracle_scores=True, oracle_costs=True
            ),
        ),
    }
    envelope = safe_envelope_bound(data, reference_bootstrap, index_sets)
    stop_gate_passed = (
        envelope["lp_upper_bound_weighted"] >= STOP_GATE_TARGET
    )
    return _round(
        {
            "report_type": REPORT_TYPE,
            "split": data.split,
            "num_episodes": data.num_episodes,
            "input_path": _repo_relative(data.input_path),
            "input_sha256": file_sha256(data.input_path),
            "outcomes_path": _repo_relative(data.outcomes_path),
            "outcomes_sha256": file_sha256(data.outcomes_path),
            "artifact_path": _repo_relative(artifact_path),
            "artifact_sha256": file_sha256(artifact_path),
            "seed": seed,
            "resamples": resamples,
            "quantile_method": "nearest-rank",
            "envelope_precision": ENVELOPE_PRECISION,
            "stop_gate_target": STOP_GATE_TARGET,
            "tier_weights": dict(TIER_WEIGHTS),
            "evidence_scope": (
                "공개 outcome을 사용하는 개발 전용 진단이며, 제출 런타임 "
                "경로에서는 절대 실행되지 않습니다."
            ),
            "reference_bootstrap": reference_bootstrap,
            "variants": variants,
            "safe_envelope_bound": envelope,
            "stop_gate_passed": stop_gate_passed,
        }
    )


def _print_report(report: Mapping[str, Any]) -> None:
    print()
    print(
        f"공개 {report['split']} {report['num_episodes']}문항 oracle 분해 "
        f"(seed {report['seed']}, {report['resamples']}회 재표본 기준)"
    )
    print(f"{'Variant':<14}" + "".join(f"{tier:>22}" for tier in TIERS) + f"{'weighted':>12}")
    for name, variant in report["variants"].items():
        cells = "".join(
            f"{variant[tier]['quality_score']:.6f}/{variant[tier]['cost_ratio']:>7.4f}".rjust(22)
            for tier in TIERS
        )
        print(f"{name:<14}{cells}{variant['weighted']:>12.6f}")
    envelope = report["safe_envelope_bound"]
    cells = "".join(
        f"{envelope[tier]['achievable_quality']:.6f}/{envelope[tier]['whole_split_cost_ratio']:>7.4f}".rjust(22)
        for tier in TIERS
    )
    print(f"{'safe-envelope':<14}{cells}{envelope['achievable_weighted']:>12.6f}")
    print()
    for tier in TIERS:
        entry = envelope[tier]
        print(
            f"  {tier:<9} 안전한도 목표 {entry['envelope_target_ratio']:.4f} "
            f"p99 {entry['resample_cost_ratio']['p99']:.4f} "
            f"(기준 {entry['reference_p99']:.4f}) "
            f"max {entry['resample_cost_ratio']['max']:.4f} "
            f"(기준 {entry['reference_max']:.4f})"
        )
    print()
    print(
        f"LP 상한 가중 점수: {envelope['lp_upper_bound_weighted']:.6f} "
        f"(정지 게이트 {report['stop_gate_target']:.6f}: "
        f"{'통과' if report['stop_gate_passed'] else '중단'})"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oracle_headroom",
        description="safe-margin 라우터의 oracle 여유 분해(개발 전용)",
    )
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = analyze_split(
            split=args.split,
            resamples=args.resamples,
            seed=args.seed,
            artifact_path=args.artifact,
            input_path=args.input,
            outcomes_path=args.outcomes,
            policy_path=args.policy,
        )
    except (
        MvpRunError,
        OSError,
        ProtocolError,
        RiskValidationError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    report_path = (
        args.report
        if args.report is not None
        else ROOT / f"build/oracle-headroom/{report['split']}.json"
    )
    write_json(report_path, report)
    if not args.quiet:
        _print_report(report)
        print(f"보고서: {report_path}")
    return 0 if report["stop_gate_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
