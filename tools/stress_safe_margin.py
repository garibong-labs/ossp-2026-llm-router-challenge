# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Development-only prompt-composition stress test for the safe-margin router.

This tool is **not** part of the submitted container runtime. It answers one
question that a single whole-split average cannot: *if the hidden evaluation
set draws the same kinds of prompts in a different mix, how far can the
realized cost ratio move?*

The method is a deterministic non-parametric bootstrap over the public split:

1. Predict every public episode exactly once with the prompt-only router, so
   the expensive feature work is not repeated per resample.
2. Draw ``--resamples`` index multisets of the same size as the split, with
   replacement, from a seeded :class:`random.Random`.
3. Re-run :func:`safe_margin.plan_selection` on each resample. The planner only
   ever sees predictions derived from prompt content, so no episode ID, split
   name, benchmark identity, input position or outcome value can reach a
   routing decision.
4. Score each resample against the public outcome matrix and record the
   distribution of the realized cost ratio per tier.

The resulting quantiles are **evidence about composition sensitivity, not a
guarantee about the hidden evaluation set**: they resample the public prompt
mix and therefore cannot describe prompts, benchmarks or token distributions
that the public split does not contain.

```console
PYTHONPATH=src python3 tools/stress_safe_margin.py --split dev
PYTHONPATH=src python3 tools/stress_safe_margin.py --split train
```
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "src"))

from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    InputBatch,
    OutcomeBatch,
    ProtocolError,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    write_json,
)

if str(ROOT / "tools") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "tools"))

from run_mvp import (  # noqa: E402  (sibling development tool)
    DEFAULT_ARTIFACT,
    MvpRunError,
    discover_split_paths,
    load_safe_margin,
)


REPORT_TYPE = "safe-margin-composition-stress-report"
STRATEGY_LABEL = "safe-margin"

#: Fixed audit configuration. The seed is recorded in every report so a reader
#: can rerun the exact same resamples.
DEFAULT_SEED = 20260822
DEFAULT_RESAMPLES = 500
#: Reported quantiles, in report order. ``max`` is always reported separately.
QUANTILES: Tuple[float, ...] = (0.5, 0.95, 0.99)
#: Nearest-rank (inclusive) quantile: ``sorted[ceil(q * n) - 1]``. Stated in the
#: report so the numbers are reproducible without reading this module.
QUANTILE_METHOD = "nearest-rank"
#: Digits kept in the JSON report, so repeated runs write identical bytes.
REPORT_DIGITS = 6


@dataclass(frozen=True)
class SplitFixture:
    """Everything one split needs, with prediction work already done once."""

    split: str
    input_path: Path
    outcomes_path: Path
    artifact_path: Path
    policy: RoutingPolicy
    predictions: Tuple[Any, ...]
    #: Realized cost per episode per model, from the public outcome matrix.
    costs: Tuple[Mapping[str, float], ...]
    #: Realized score per episode per model, from the public outcome matrix.
    scores: Tuple[Mapping[str, float], ...]

    @property
    def num_episodes(self) -> int:
        return len(self.predictions)


def outcome_cost(outcome: Any, policy: RoutingPolicy) -> Decimal:
    """Return the official v1 cost of one outcome row.

    This repeats the formula used by :func:`ossp_router.scoring.score_submissions`
    rather than importing a private helper. ``tests/test_stress_safe_margin.py``
    pins the whole-split ratio against the official scorer, so a divergence
    here fails the suite instead of silently changing the audit.
    """

    rates = policy.models[outcome.model_id]
    unit = Decimal(policy.token_unit)
    return (
        rates.fixed_cost
        + Decimal(outcome.input_tokens) * rates.input_token_rate / unit
        + Decimal(outcome.output_tokens) * rates.output_token_rate / unit
    )


def outcome_tables(
    inputs: InputBatch, outcomes: OutcomeBatch, policy: RoutingPolicy
) -> Tuple[Tuple[Mapping[str, float], ...], Tuple[Mapping[str, float], ...]]:
    """Build per-episode realized cost and score tables in input order."""

    by_key = {
        (item.episode_id, item.model_id): item for item in outcomes.outcomes
    }
    costs: List[Mapping[str, float]] = []
    scores: List[Mapping[str, float]] = []
    for episode in inputs.episodes:
        try:
            rows = {
                model_id: by_key[(episode.episode_id, model_id)]
                for model_id in MODEL_IDS
            }
        except KeyError as exc:  # pragma: no cover - guarded by the scorer too
            raise MvpRunError(
                f"outcome 행렬에 누락된 항목이 있습니다: {exc}"
            ) from exc
        cost_row = {
            model_id: float(outcome_cost(row, policy))
            for model_id, row in rows.items()
        }
        if cost_row[policy.light_model_id] <= 0:
            raise MvpRunError(
                f"light 모델 실제 비용이 0 이하입니다: {episode.episode_id}"
            )
        costs.append(cost_row)
        scores.append(
            {model_id: float(row.score) for model_id, row in rows.items()}
        )
    return tuple(costs), tuple(scores)


def prepare_split(
    *,
    split: str = "dev",
    input_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    artifact_path: Path = DEFAULT_ARTIFACT,
    policy_path: Optional[Path] = None,
) -> SplitFixture:
    """Load one public split and predict every episode exactly once."""

    safe_margin = load_safe_margin()
    paths = discover_split_paths(split, input_path, outcomes_path)
    inputs = load_input(paths["input"])
    outcomes = load_outcomes(paths["outcomes"])
    policy = (
        load_policy(policy_path) if policy_path is not None else load_bundled_policy()
    )
    artifact = safe_margin.load_artifact(artifact_path)
    if artifact.policy_id != policy.policy_id:
        raise MvpRunError("artifact와 정책의 policy_id가 다릅니다.")
    predictions = safe_margin.predict_batch(inputs.episodes, artifact, policy)
    costs, scores = outcome_tables(inputs, outcomes, policy)
    return SplitFixture(
        split=inputs.split,
        input_path=paths["input"],
        outcomes_path=paths["outcomes"],
        artifact_path=artifact_path,
        policy=policy,
        predictions=predictions,
        costs=costs,
        scores=scores,
    )


def bootstrap_indices(
    sample_size: int, resamples: int, seed: int
) -> Tuple[Tuple[int, ...], ...]:
    """Draw deterministic index multisets, with replacement.

    The same index sets are reused for every tier so the three tier
    distributions describe the *same* hypothetical prompt mixes.
    """

    if sample_size <= 0:
        raise ValueError("표본 크기는 1 이상이어야 합니다.")
    if resamples <= 0:
        raise ValueError("재표본 수는 1 이상이어야 합니다.")
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(sample_size) for _position in range(sample_size))
        for _resample in range(resamples)
    )


def quantile(ordered: Sequence[float], q: float) -> float:
    """Return the inclusive nearest-rank quantile of an ascending sequence."""

    if not ordered:
        raise ValueError("분위수를 계산할 값이 없습니다.")
    rank = int(math.ceil(q * len(ordered)))
    return ordered[min(len(ordered) - 1, max(1, rank) - 1)]


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(values)
    summary = {
        "mean": math.fsum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    for q in QUANTILES:
        summary[f"p{int(round(q * 100))}"] = quantile(ordered, q)
    return summary


def evaluate_selection(
    fixture: SplitFixture,
    indices: Sequence[int],
    selected: Sequence[str],
) -> Tuple[float, float]:
    """Return the realized cost ratio and mean quality of one plan."""

    light_id = fixture.policy.light_model_id
    total = math.fsum(
        fixture.costs[index][model_id]
        for index, model_id in zip(indices, selected)
    )
    light_total = math.fsum(fixture.costs[index][light_id] for index in indices)
    quality = math.fsum(
        fixture.scores[index][model_id]
        for index, model_id in zip(indices, selected)
    )
    return total / light_total, quality / len(indices)


def plan_indices(
    fixture: SplitFixture,
    indices: Sequence[int],
    tier: str,
    config: Optional[Any] = None,
) -> Tuple[str, ...]:
    """Run the prompt-only planner over one index multiset."""

    safe_margin = sys.modules["safe_margin"]
    subset = [fixture.predictions[index] for index in indices]
    selected, _ratio, _stages = safe_margin.plan_selection(
        subset, fixture.policy, tier, config
    )
    return selected


def step_multiples(
    fixture: SplitFixture,
    indices: Sequence[int],
    selected: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Realized per-episode cost of the chosen model over its light cost.

    This is the tail that a whole-split average hides: one selected episode
    whose upgraded generation cost dozens of times its light generation moves
    the ratio of any resample that happens to draw it more than once.
    """

    light_id = fixture.policy.light_model_id
    per_model: Dict[str, List[float]] = {model_id: [] for model_id in MODEL_IDS}
    for index, model_id in zip(indices, selected):
        row = fixture.costs[index]
        per_model[model_id].append(row[model_id] / row[light_id])
    return {
        model_id: _distribution(values)
        for model_id, values in per_model.items()
        if values
    }


def stress_tier(
    fixture: SplitFixture,
    tier: str,
    index_sets: Sequence[Sequence[int]],
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run every resample through one tier and summarize the distribution."""

    if not index_sets:
        raise ValueError("재표본 색인 집합은 비어 있을 수 없습니다.")
    cap = float(fixture.policy.tiers[tier].budget_multiplier)
    ratios: List[float] = []
    qualities: List[float] = []
    breaches = 0
    worst: Dict[str, Any] = {}
    for resample, indices in enumerate(index_sets):
        selected = plan_indices(fixture, indices, tier, config)
        ratio, quality = evaluate_selection(fixture, indices, selected)
        ratios.append(ratio)
        qualities.append(quality)
        if ratio > cap:
            breaches += 1
        if not worst or ratio > worst["cost_ratio"]:
            worst = {
                "resample": resample,
                "cost_ratio": ratio,
                "quality_score": quality,
            }
    return {
        "budget_multiplier": cap,
        "resamples": len(index_sets),
        "cost_ratio": _distribution(ratios),
        "quality_score": _distribution(qualities),
        "cap_breaches": breaches,
        "cap_breach_rate": breaches / len(index_sets),
        "worst_resample": worst,
    }


def whole_split_tier(
    fixture: SplitFixture, tier: str, config: Optional[Any] = None
) -> Dict[str, Any]:
    """Plan and score the split exactly once, without resampling."""

    indices = tuple(range(fixture.num_episodes))
    selected = plan_indices(fixture, indices, tier, config)
    ratio, quality = evaluate_selection(fixture, indices, selected)
    cap = float(fixture.policy.tiers[tier].budget_multiplier)
    warning = float(fixture.policy.budget_warning_ratio)
    return {
        "budget_multiplier": cap,
        "cost_ratio": ratio,
        "quality_score": quality,
        "budget_passed": ratio <= cap,
        "near_budget": ratio >= cap * warning,
        "model_counts": {
            model_id: sum(1 for item in selected if item == model_id)
            for model_id in MODEL_IDS
        },
        "selected_step_multiple": step_multiples(fixture, indices, selected),
    }


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, REPORT_DIGITS)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def stress_split(
    *,
    split: str = "dev",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    input_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    artifact_path: Path = DEFAULT_ARTIFACT,
    policy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return the full deterministic stress report for one public split."""

    fixture = prepare_split(
        split=split,
        input_path=input_path,
        outcomes_path=outcomes_path,
        artifact_path=artifact_path,
        policy_path=policy_path,
    )
    index_sets = bootstrap_indices(fixture.num_episodes, resamples, seed)
    report = {
        "report_type": REPORT_TYPE,
        "strategy": STRATEGY_LABEL,
        "schema_version": fixture.policy.schema_version,
        "policy_id": fixture.policy.policy_id,
        "split": fixture.split,
        "num_episodes": fixture.num_episodes,
        "input_path": _repo_relative(fixture.input_path),
        "outcomes_path": _repo_relative(fixture.outcomes_path),
        "artifact_path": _repo_relative(fixture.artifact_path),
        "seed": seed,
        "resamples": resamples,
        "sample_size": fixture.num_episodes,
        "with_replacement": True,
        "quantile_method": QUANTILE_METHOD,
        "quantiles": list(QUANTILES),
        "evidence_scope": (
            "공개 split 프롬프트 구성을 재표본한 근거이며, 비공개 평가셋의 "
            "예산 통과를 보장하지 않습니다."
        ),
        "whole_split": {
            tier: whole_split_tier(fixture, tier) for tier in TIERS
        },
        "bootstrap": {
            tier: stress_tier(fixture, tier, index_sets) for tier in TIERS
        },
    }
    report["total_cap_breaches"] = sum(
        report["bootstrap"][tier]["cap_breaches"] for tier in TIERS
    )
    return _round(report)


def _print_report(report: Mapping[str, Any]) -> None:
    print()
    print(
        f"공개 {report['split']} {report['num_episodes']}문항 구성 스트레스 "
        f"(seed {report['seed']}, {report['resamples']}회 재표본, "
        f"{report['quantile_method']})"
    )
    header = f"{'Tier':<10}{'한도':>7}{'전체':>11}"
    for label in ("mean", "p50", "p95", "p99", "max"):
        header += f"{label:>11}"
    header += f"{'초과':>7}"
    print(header)
    print("-" * len(header))
    for tier in TIERS:
        whole = report["whole_split"][tier]
        stats = report["bootstrap"][tier]["cost_ratio"]
        row = f"{tier:<10}{whole['budget_multiplier']:>7}{whole['cost_ratio']:>11.4f}"
        for label in ("mean", "p50", "p95", "p99", "max"):
            row += f"{stats[label]:>11.4f}"
        row += f"{report['bootstrap'][tier]['cap_breaches']:>7}"
        print(row)
    print()
    print("등급별 실제 품질과 선택 분포")
    for tier in TIERS:
        whole = report["whole_split"][tier]
        counts = ", ".join(
            f"{model_id}={whole['model_counts'][model_id]}" for model_id in MODEL_IDS
        )
        print(
            f"  {tier:<9} 품질 {whole['quality_score']:.6f} "
            f"near_budget={whole['near_budget']} {counts}"
        )
        for model_id, stats in whole["selected_step_multiple"].items():
            print(
                f"  {'':<9}   선택 {model_id:<12} 실제 비용 배수 "
                f"p95={stats['p95']:.2f} p99={stats['p99']:.2f} "
                f"max={stats['max']:.2f}"
            )
    print()
    print(
        "이 수치는 공개 자료 구성 민감도에 대한 근거이며, "
        "비공개 평가셋 안전을 보장하지 않습니다."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stress_safe_margin",
        description="safe-margin 라우터의 결정적 구성 스트레스 검사(개발 전용)",
    )
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = stress_split(
            split=args.split,
            resamples=args.resamples,
            seed=args.seed,
            input_path=args.input,
            outcomes_path=args.outcomes,
            artifact_path=args.artifact,
            policy_path=args.policy,
        )
    except (
        MvpRunError,
        OSError,
        ProtocolError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    report_path = (
        args.report
        if args.report is not None
        else ROOT / f"build/stress/{report['split']}.json"
    )
    write_json(report_path, report)
    if not args.quiet:
        _print_report(report)
        print(f"보고서: {report_path}")
    if report["total_cap_breaches"]:
        print(
            f"오류: 재표본에서 한도를 넘은 사례가 있습니다 "
            f"({report['total_cap_breaches']}건).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
