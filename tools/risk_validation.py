# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Development-only comparable risk-validation harness for router candidates.

This tool is **not** part of the submitted container runtime. It measures the
budget-overrun tail of one or two prompt-only routing policies under identical
deterministic evidence and decides a frozen safety gate:

* 5,000 deterministic prompt-composition bootstrap resamples per tier, shared
  across every policy in the same run, so the distributions describe the same
  hypothetical prompt mixes;
* source/task-family grouped holdouts. Family labels are reconstructed only
  from the public materialization inputs (the AIME and DeepMind-Mathematics
  selection files plus deterministic prompt-content rules); no outcome value
  participates in a label and no label ever reaches a runtime decision;
* per tier: whole-split ratio, resample mean/p50/p95/p99/max (nearest-rank),
  hard-cap breach count, model counts and single-episode cost concentration.

The candidate gate is evaluated against the *same-run* safe-margin reference:

* breach count must be ``0`` out of all resamples for every tier;
* the candidate's p99 and maximum resample cost ratios may not exceed the
  reference by more than :data:`HEADROOM_TOLERANCE` (documented tolerance for
  numerically-close distributions, not a budget expansion);
* every family holdout must stay under the hard cap and may not exceed the
  reference holdout ratio by more than the same tolerance; a family that
  cannot be evaluated makes the gate *indeterminate*, which fails.

```console
PYTHONPATH=src python3 tools/risk_validation.py --split dev
PYTHONPATH=src python3 tools/risk_validation.py --split train --split dev \
  --candidate-artifact baselines/risk-calibrated-public.v2.json
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "tools") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "tools"))

from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    InputBatch,
    ProtocolError,
    RoutingPolicy,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    write_json,
)

from run_mvp import (  # noqa: E402  (sibling development tool)
    DEFAULT_ARTIFACT,
    MvpRunError,
    discover_split_paths,
    load_safe_margin,
)
from stress_safe_margin import (  # noqa: E402  (sibling development tool)
    bootstrap_indices,
    outcome_tables,
    quantile,
)


REPORT_TYPE = "router-risk-validation-report"

#: Frozen audit configuration for the v2 safety gate. The seed differs from the
#: older 500-resample stress report on purpose: the gate is defined against a
#: newly measured 5,000-resample reference, not the historical table.
DEFAULT_SEED = 20260825
DEFAULT_RESAMPLES = 5000
#: Absolute realized-cost-ratio tolerance when comparing a candidate's p99/max
#: (and family-holdout ratios) against the same-run safe-margin reference.
#: This absorbs numerically-close distributions only; it is far smaller than
#: any tier's distance to its hard cap.
HEADROOM_TOLERANCE = 0.005
QUANTILE_METHOD = "nearest-rank"
REPORT_DIGITS = 6

#: Task families reconstructable from public materialization inputs. AIME and
#: DeepMind Mathematics come from the public selection files; the remaining
#: labels use deterministic prompt-content rules. Labels are development-only
#: grouping evidence: they never reach a runtime decision path.
FAMILY_LABELS = (
    "aime",
    "babilong",
    "belebele-ko",
    "cruxeval",
    "deepmind-mathematics",
    "gsm8k",
    "hrmcr",
    "ruletaker",
    "truthfulqa",
)

_RULE_SENTENCE = re.compile(
    r"(?:^|\. )(?:If |[A-Z][a-z]+ (?:is|does not|likes|visits|chases|eats|"
    r"needs|sees)\b|The [a-z ]+ (?:is|does not|likes|visits|chases|eats|"
    r"needs|sees)\b)"
)
_CRUX_ASSERT = re.compile(r"assert f\(")


class RiskValidationError(RuntimeError):
    """Raised when the harness cannot produce a comparable measurement."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_prompt(prompt: str) -> str:
    """Classify one prompt into a content-derived task family.

    Only used for development-time grouped holdouts. The AIME and
    DeepMind-Mathematics rows are labeled from selection files before this
    function is consulted, so this only distinguishes the redistributable
    sources.
    """

    nonspace = sum(not character.isspace() for character in prompt)
    hangul = sum("가" <= character <= "힣" for character in prompt)
    hangul_ratio = hangul / max(1, nonspace)
    if _CRUX_ASSERT.search(prompt) and "def f(" in prompt:
        return "cruxeval"
    if len(prompt) >= 3000:
        return "babilong"
    if prompt.startswith("Question:"):
        return "truthfulqa"
    if hangul_ratio > 0.2 and "\nQuestion:" in prompt and "\nA." in prompt:
        return "belebele-ko"
    if hangul_ratio > 0.2:
        return "hrmcr"
    if len(_RULE_SENTENCE.findall(prompt)) >= 4:
        return "ruletaker"
    return "gsm8k"


def _selection_ids(path: Path, label: str) -> frozenset:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RiskValidationError(f"{label} 선택 파일을 읽을 수 없습니다: {exc}") from exc
    episodes = value.get("episodes") if isinstance(value, dict) else None
    if not isinstance(episodes, list):
        raise RiskValidationError(f"{label} 선택 파일 구조가 올바르지 않습니다.")
    return frozenset(str(row.get("episode_id")) for row in episodes)


def _deepmind_ids(split: str) -> frozenset:
    path = ROOT / "data/sources/deepmind-mathematics-selection.v1.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value["splits"][split]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RiskValidationError(
            f"deepmind-mathematics 선택 파일을 읽을 수 없습니다: {exc}"
        ) from exc
    return frozenset(str(row.get("episode_id")) for row in rows)


def reconstruct_families(split: str, inputs: InputBatch) -> Tuple[str, ...]:
    """Label every episode with a deterministic source/task family."""

    aime = _selection_ids(ROOT / f"data/{split}/aime-selection.json", split)
    deepmind = _deepmind_ids(split)
    labels: List[str] = []
    for episode in inputs.episodes:
        if episode.episode_id in aime:
            labels.append("aime")
        elif episode.episode_id in deepmind:
            labels.append("deepmind-mathematics")
        else:
            if episode.prompt is None:
                raise RiskValidationError(
                    "공개 자료 문항에는 prompt가 있어야 합니다."
                )
            labels.append(classify_prompt(episode.prompt))
    return tuple(labels)


@dataclass(frozen=True)
class SplitData:
    """One public split with realized outcome tables and family labels."""

    split: str
    input_path: Path
    outcomes_path: Path
    policy: RoutingPolicy
    inputs: InputBatch
    costs: Tuple[Mapping[str, float], ...]
    scores: Tuple[Mapping[str, float], ...]
    families: Tuple[str, ...]

    @property
    def num_episodes(self) -> int:
        return len(self.inputs.episodes)


def load_split_data(
    split: str,
    *,
    input_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    policy_path: Optional[Path] = None,
) -> SplitData:
    paths = discover_split_paths(split, input_path, outcomes_path)
    inputs = load_input(paths["input"])
    outcomes = load_outcomes(paths["outcomes"])
    policy = (
        load_policy(policy_path) if policy_path is not None else load_bundled_policy()
    )
    costs, scores = outcome_tables(inputs, outcomes, policy)
    families = reconstruct_families(inputs.split, inputs)
    return SplitData(
        split=inputs.split,
        input_path=paths["input"],
        outcomes_path=paths["outcomes"],
        policy=policy,
        inputs=inputs,
        costs=costs,
        scores=scores,
        families=families,
    )


@dataclass(frozen=True)
class PolicyRunner:
    """A prompt-only policy prepared for one split.

    ``plan`` receives an index multiset over the split's episodes plus a tier
    and returns one model per drawn index. Predictions are computed exactly
    once per split from prompt content, so no episode ID, position, family
    label or outcome value can reach a decision.
    """

    name: str
    artifact_path: Path
    artifact_sha256: str
    plan: Callable[[Sequence[int], str], Tuple[str, ...]]


def safe_margin_runner(
    data: SplitData, artifact_path: Path = DEFAULT_ARTIFACT
) -> PolicyRunner:
    safe_margin = load_safe_margin()
    artifact = safe_margin.load_artifact(artifact_path)
    predictions = safe_margin.predict_batch(
        data.inputs.episodes, artifact, data.policy
    )

    def plan(indices: Sequence[int], tier: str) -> Tuple[str, ...]:
        subset = [predictions[index] for index in indices]
        selected, _ratio, _stages = safe_margin.plan_selection(
            subset, data.policy, tier
        )
        return selected

    return PolicyRunner(
        name="safe-margin",
        artifact_path=artifact_path,
        artifact_sha256=file_sha256(artifact_path),
        plan=plan,
    )


def risk_calibrated_runner(data: SplitData, artifact_path: Path) -> PolicyRunner:
    """Prepare the risk-calibrated v2 candidate for one split."""

    import importlib.util

    if "risk_calibrated" in sys.modules:
        risk_calibrated = sys.modules["risk_calibrated"]
    else:
        load_safe_margin()  # ensures the sibling hash_regex import path works
        spec = importlib.util.spec_from_file_location(
            "risk_calibrated", ROOT / "baselines/risk_calibrated.py"
        )
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise RiskValidationError("risk_calibrated 모듈을 불러올 수 없습니다.")
        risk_calibrated = importlib.util.module_from_spec(spec)
        sys.modules["risk_calibrated"] = risk_calibrated
        spec.loader.exec_module(risk_calibrated)
    artifact = risk_calibrated.load_artifact(artifact_path)
    predictions = risk_calibrated.predict_batch(
        data.inputs.episodes, artifact, data.policy
    )

    def plan(indices: Sequence[int], tier: str) -> Tuple[str, ...]:
        subset = [predictions[index] for index in indices]
        selected, _ratio, _stages = risk_calibrated.plan_selection(
            subset, data.policy, tier, artifact.tier_plans
        )
        return selected

    return PolicyRunner(
        name="risk-calibrated-v2",
        artifact_path=artifact_path,
        artifact_sha256=file_sha256(artifact_path),
        plan=plan,
    )


def realized_ratio_quality(
    data: SplitData, indices: Sequence[int], selected: Sequence[str]
) -> Tuple[float, float]:
    light_id = data.policy.light_model_id
    total = math.fsum(
        data.costs[index][model_id] for index, model_id in zip(indices, selected)
    )
    light_total = math.fsum(data.costs[index][light_id] for index in indices)
    quality = math.fsum(
        data.scores[index][model_id] for index, model_id in zip(indices, selected)
    )
    return total / light_total, quality / len(indices)


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(values)
    result = {
        "mean": math.fsum(ordered) / len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }
    for q in (0.5, 0.95, 0.99):
        result[f"p{int(round(q * 100))}"] = quantile(ordered, q)
    return result


def _concentration(
    data: SplitData, indices: Sequence[int], selected: Sequence[str]
) -> float:
    """Largest single-episode share of the realized discretionary spend."""

    light_id = data.policy.light_model_id
    increments = [
        data.costs[index][model_id] - data.costs[index][light_id]
        for index, model_id in zip(indices, selected)
    ]
    total = math.fsum(increments)
    if total <= 0:
        return 0.0
    return max(increments) / total


def whole_split_metrics(
    data: SplitData, runner: PolicyRunner, tier: str
) -> Dict[str, Any]:
    indices = tuple(range(data.num_episodes))
    selected = runner.plan(indices, tier)
    ratio, quality = realized_ratio_quality(data, indices, selected)
    cap = float(data.policy.tiers[tier].budget_multiplier)
    return {
        "budget_multiplier": cap,
        "cost_ratio": ratio,
        "quality_score": quality,
        "budget_passed": ratio <= cap,
        "model_counts": {
            model_id: sum(1 for item in selected if item == model_id)
            for model_id in MODEL_IDS
        },
        "single_episode_concentration": _concentration(data, indices, selected),
    }


def bootstrap_metrics(
    data: SplitData,
    runner: PolicyRunner,
    tier: str,
    index_sets: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    cap = float(data.policy.tiers[tier].budget_multiplier)
    ratios: List[float] = []
    qualities: List[float] = []
    breaches = 0
    worst: Dict[str, Any] = {}
    for resample, indices in enumerate(index_sets):
        selected = runner.plan(indices, tier)
        ratio, quality = realized_ratio_quality(data, indices, selected)
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


def family_holdout_metrics(
    data: SplitData, runner: PolicyRunner, tier: str
) -> Dict[str, Any]:
    """Plan the split with one whole source family removed at a time."""

    cap = float(data.policy.tiers[tier].budget_multiplier)
    result: Dict[str, Any] = {}
    for family in FAMILY_LABELS:
        indices = tuple(
            index
            for index in range(data.num_episodes)
            if data.families[index] != family
        )
        if not indices or len(indices) == data.num_episodes:
            result[family] = {"evaluated": False, "held_out": 0}
            continue
        selected = runner.plan(indices, tier)
        ratio, quality = realized_ratio_quality(data, indices, selected)
        result[family] = {
            "evaluated": True,
            "held_out": data.num_episodes - len(indices),
            "cost_ratio": ratio,
            "quality_score": quality,
            "budget_passed": ratio <= cap,
            "single_episode_concentration": _concentration(
                data, indices, selected
            ),
        }
    return result


def evaluate_runner(
    data: SplitData,
    runner: PolicyRunner,
    index_sets: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    return {
        "policy": runner.name,
        "artifact_path": _repo_relative(runner.artifact_path),
        "artifact_sha256": runner.artifact_sha256,
        "tiers": {
            tier: {
                "whole_split": whole_split_metrics(data, runner, tier),
                "bootstrap": bootstrap_metrics(data, runner, tier, index_sets),
                "family_holdouts": family_holdout_metrics(data, runner, tier),
            }
            for tier in TIERS
        },
    }


def gate_candidate(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    tolerance: float = HEADROOM_TOLERANCE,
) -> Dict[str, Any]:
    """Decide the frozen safety gate for one split's paired measurement."""

    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    for tier in TIERS:
        ref_tier = reference["tiers"][tier]
        cand_tier = candidate["tiers"][tier]
        breaches = cand_tier["bootstrap"]["cap_breaches"]
        check(
            f"{tier}.bootstrap.breaches",
            breaches == 0,
            f"candidate breaches={breaches}",
        )
        for stat in ("p99", "max"):
            ref_value = ref_tier["bootstrap"]["cost_ratio"][stat]
            cand_value = cand_tier["bootstrap"]["cost_ratio"][stat]
            check(
                f"{tier}.bootstrap.{stat}",
                cand_value <= ref_value + tolerance,
                f"candidate {cand_value:.6f} vs reference {ref_value:.6f} "
                f"(+{tolerance})",
            )
        check(
            f"{tier}.whole_split.budget",
            cand_tier["whole_split"]["budget_passed"],
            f"ratio {cand_tier['whole_split']['cost_ratio']:.6f}",
        )
        for family in FAMILY_LABELS:
            ref_family = ref_tier["family_holdouts"][family]
            cand_family = cand_tier["family_holdouts"][family]
            if not ref_family.get("evaluated") or not cand_family.get("evaluated"):
                check(
                    f"{tier}.holdout.{family}",
                    False,
                    "indeterminate: family holdout could not be evaluated",
                )
                continue
            ok = (
                cand_family["budget_passed"]
                and cand_family["cost_ratio"]
                <= ref_family["cost_ratio"] + tolerance
            )
            check(
                f"{tier}.holdout.{family}",
                ok,
                f"candidate {cand_family['cost_ratio']:.6f} vs reference "
                f"{ref_family['cost_ratio']:.6f} (+{tolerance})",
            )
    return {
        "tolerance": tolerance,
        "passed": all(item["passed"] for item in checks),
        "failed_checks": [item for item in checks if not item["passed"]],
        "checks": checks,
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
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def validate_split(
    *,
    split: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    candidate_artifact: Optional[Path] = None,
    tolerance: float = HEADROOM_TOLERANCE,
    input_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    policy_path: Optional[Path] = None,
    reference_artifact: Path = DEFAULT_ARTIFACT,
) -> Dict[str, Any]:
    """Measure the reference (and optionally one candidate) on one split."""

    data = load_split_data(
        split,
        input_path=input_path,
        outcomes_path=outcomes_path,
        policy_path=policy_path,
    )
    index_sets = bootstrap_indices(data.num_episodes, resamples, seed)
    reference = safe_margin_runner(data, reference_artifact)
    report: Dict[str, Any] = {
        "report_type": REPORT_TYPE,
        "schema_version": data.policy.schema_version,
        "policy_id": data.policy.policy_id,
        "split": data.split,
        "num_episodes": data.num_episodes,
        "input_path": _repo_relative(data.input_path),
        "input_sha256": file_sha256(data.input_path),
        "outcomes_path": _repo_relative(data.outcomes_path),
        "outcomes_sha256": file_sha256(data.outcomes_path),
        "seed": seed,
        "resamples": resamples,
        "sample_size": data.num_episodes,
        "with_replacement": True,
        "quantile_method": QUANTILE_METHOD,
        "family_counts": {
            family: data.families.count(family) for family in FAMILY_LABELS
        },
        "evidence_scope": (
            "공개 split 프롬프트 구성을 재표본한 근거이며, 비공개 평가셋의 "
            "예산 통과를 보장하지 않습니다."
        ),
        "reference": evaluate_runner(data, reference, index_sets),
    }
    if candidate_artifact is not None:
        candidate = risk_calibrated_runner(data, candidate_artifact)
        report["candidate"] = evaluate_runner(data, candidate, index_sets)
        report["gate"] = gate_candidate(
            report["reference"], report["candidate"], tolerance=tolerance
        )
    return _round(report)


def _print_split_report(report: Mapping[str, Any]) -> None:
    print()
    print(
        f"공개 {report['split']} {report['num_episodes']}문항 위험 검증 "
        f"(seed {report['seed']}, {report['resamples']}회 재표본)"
    )
    for label in ("reference", "candidate"):
        if label not in report:
            continue
        entry = report[label]
        print(f"  [{entry['policy']}]")
        for tier in TIERS:
            tier_report = entry["tiers"][tier]
            whole = tier_report["whole_split"]
            stats = tier_report["bootstrap"]["cost_ratio"]
            print(
                f"    {tier:<9} 전체 {whole['cost_ratio']:.4f} "
                f"품질 {whole['quality_score']:.6f} "
                f"p99 {stats['p99']:.4f} max {stats['max']:.4f} "
                f"초과 {tier_report['bootstrap']['cap_breaches']}"
                f"/{tier_report['bootstrap']['resamples']} "
                f"집중도 {whole['single_episode_concentration']:.3f}"
            )
    if "gate" in report:
        gate = report["gate"]
        print(f"  gate: {'PASS' if gate['passed'] else 'FAIL'}")
        for item in gate["failed_checks"]:
            print(f"    실패: {item['check']}: {item['detail']}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="risk_validation",
        description="후보 라우터의 결정적 위험 검증 하네스(개발 전용)",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "dev"),
        default=[],
        help="검증할 공개 split (여러 번 지정 가능, 기본 train+dev)",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tolerance", type=float, default=HEADROOM_TOLERANCE)
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--reference-artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    splits = args.split or ["train", "dev"]
    reports = {}
    try:
        for split in splits:
            reports[split] = validate_split(
                split=split,
                resamples=args.resamples,
                seed=args.seed,
                candidate_artifact=args.candidate_artifact,
                tolerance=args.tolerance,
                policy_path=args.policy,
                reference_artifact=args.reference_artifact,
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
    combined = {
        "report_type": REPORT_TYPE,
        "seed": args.seed,
        "resamples": args.resamples,
        "tolerance": args.tolerance,
        "splits": reports,
    }
    if args.candidate_artifact is not None:
        combined["gate_passed"] = all(
            reports[split].get("gate", {}).get("passed", False)
            for split in splits
        )
    report_path = (
        args.report
        if args.report is not None
        else ROOT / "build/risk-validation/report.json"
    )
    write_json(report_path, combined)
    if not args.quiet:
        for split in splits:
            _print_split_report(reports[split])
        print(f"보고서: {report_path}")
    if args.candidate_artifact is not None and not combined["gate_passed"]:
        print("오류: 후보가 안전 게이트를 통과하지 못했습니다.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
