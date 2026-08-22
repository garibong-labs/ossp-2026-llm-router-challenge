# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Development-only one-command build and self-check for the MVP router.

This tool is **not** part of the submitted container runtime. It generates the
three safe-margin tier submissions for one public split, runs the official
scoring path from :mod:`ossp_router.scoring`, and prints a compact comparison
against the published baselines.

```console
PYTHONPATH=src python3 tools/run_mvp.py
```
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # pragma: no cover - convenience path
    sys.path.insert(0, str(ROOT / "src"))

from ossp_router.heuristic import write_submission_atomic  # noqa: E402
from ossp_router.protocol import (  # noqa: E402
    MODEL_IDS,
    TIERS,
    ProtocolError,
    load_bundled_policy,
    load_input,
    load_outcomes,
    load_policy,
    write_json,
)
from ossp_router.scoring import ScoringError, score_submissions  # noqa: E402


DEFAULT_ARTIFACT = ROOT / "baselines/hash-regex-public.v1.json"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "build/mvp"
STRATEGY_LABEL = "safe-margin"

#: Published public Dev figures from `baselines/README.md`, used only to print
#: the comparison table. Each entry is (quality score, actual cost ratio).
PUBLIC_DEV_BASELINES: Mapping[str, Mapping[str, Sequence[str]]] = {
    "all-light": {
        "fast": ("0.619318", "1.000000"),
        "balanced": ("0.619318", "1.000000"),
        "premium": ("0.619318", "1.000000"),
        "final": ("0.619318", ""),
    },
    "prompt-heuristic": {
        "fast": ("0.625852", "1.072334"),
        "balanced": ("0.658239", "1.367866"),
        "premium": ("0.691761", "2.102044"),
        "final": ("0.655341", ""),
    },
    "feature-budget": {
        "fast": ("0.621023", "1.038210"),
        "balanced": ("0.623580", "1.334059"),
        "premium": ("0.691761", "2.102044"),
        "final": ("0.643011", ""),
    },
    "hash-regex": {
        "fast": ("0.663068", "1.235989"),
        "balanced": ("0.693750", "1.961506"),
        "premium": ("0.740057", "3.985205"),
        "final": ("0.695369", ""),
    },
}


class MvpRunError(RuntimeError):
    """Raised when the development runner cannot complete a stage."""


def _load_baseline_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise MvpRunError(f"{path} 모듈을 불러올 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_safe_margin():
    """Import the sibling baseline modules without installing a package."""

    _load_baseline_module("hash_regex", ROOT / "baselines/hash_regex.py")
    return _load_baseline_module("safe_margin", ROOT / "baselines/safe_margin.py")


def discover_split_paths(
    split: str,
    input_path: Optional[Path],
    outcomes_path: Optional[Path],
) -> Dict[str, Path]:
    """Resolve the materialized public input and the public outcome file."""

    resolved_input = (
        input_path
        if input_path is not None
        else ROOT / f"data/materialized/{split}/inputs.json"
    )
    resolved_outcomes = (
        outcomes_path
        if outcomes_path is not None
        else ROOT / f"data/{split}/outcomes.json"
    )
    if not resolved_input.is_file():
        raise MvpRunError(
            f"공개 입력 파일이 없습니다: {resolved_input}\n"
            "먼저 tools/materialize_public_data.py로 공개 자료를 준비하십시오."
        )
    if not resolved_outcomes.is_file():
        raise MvpRunError(f"공개 outcome 파일이 없습니다: {resolved_outcomes}")
    return {"input": resolved_input, "outcomes": resolved_outcomes}


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path so reports carry no local absolute path."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _format_row(label: str, cells: Sequence[str]) -> str:
    return f"{label:<18}" + "".join(f"{cell:>26}" for cell in cells)


def _print_report(report: Mapping[str, Any], plans: Mapping[str, Any]) -> None:
    print()
    print(f"공개 {report['split']} {report['tiers']['fast']['num_episodes']}문항 결과")
    print(_format_row("Router", [*(tier.capitalize() for tier in TIERS), "Weighted"]))
    print("-" * (18 + 26 * 4))
    for name, values in PUBLIC_DEV_BASELINES.items():
        cells = [f"{values[tier][0]} / {values[tier][1]}" for tier in TIERS]
        cells.append(values["final"][0])
        print(_format_row(name, cells))
    cells = []
    for tier in TIERS:
        entry = report["tiers"][tier]
        marker = "" if entry["budget_passed"] else "  OVER-BUDGET"
        cells.append(f"{entry['quality_score'][:8]} / {entry['budget_ratio'][:8]}{marker}")
    cells.append(report["final_score"][:8])
    print(_format_row(f"{STRATEGY_LABEL} (MVP)", cells))

    print()
    print("등급별 상세")
    for tier in TIERS:
        entry = report["tiers"][tier]
        plan = plans[tier]
        counts = ", ".join(
            f"{model_id}={entry['model_counts'][model_id]}" for model_id in MODEL_IDS
        )
        print(
            f"  {tier:<9} 비용 비율 {entry['budget_ratio']:<16} "
            f"한도 {entry['budget_multiplier']:<5} "
            f"안전 목표 {plan.target_ratio:<5} "
            f"예산통과={entry['budget_passed']} near_budget={entry['near_budget']}"
        )
        print(
            f"  {'':<9} 품질 {entry['quality_score']:<16} "
            f"예측 비용 비율 {plan.predicted_budget_ratio:.6f}"
        )
        print(f"  {'':<9} 선택 분포 {counts}")
        for stage in plan.stages:
            print(
                f"  {'':<9}   stage {stage.step:<22} "
                f"eligible={stage.eligible:<6} promoted={stage.promoted:<6} "
                f"groups={stage.groups_promoted}/{stage.groups_considered}"
            )
    print()
    print(f"weighted final score: {report['final_score']}")


def run_mvp(
    *,
    split: str = "dev",
    input_path: Optional[Path] = None,
    outcomes_path: Optional[Path] = None,
    artifact_path: Path = DEFAULT_ARTIFACT,
    policy_path: Optional[Path] = None,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Generate the three tier submissions and score them with the official path."""

    safe_margin = load_safe_margin()
    paths = discover_split_paths(split, input_path, outcomes_path)
    inputs = load_input(paths["input"])
    outcomes = load_outcomes(paths["outcomes"])
    policy = (
        load_policy(policy_path) if policy_path is not None else load_bundled_policy()
    )
    artifact = safe_margin.load_artifact(artifact_path)

    plans = {}
    submissions = []
    for tier in TIERS:
        plan = safe_margin.make_safe_margin_submission(inputs, policy, artifact, tier)
        write_submission_atomic(output_directory / f"{tier}.json", plan.submission)
        plans[tier] = plan
        submissions.append(plan.submission)

    report = score_submissions(inputs, outcomes, submissions, policy)
    report_path = output_directory / "report.json"
    write_json(
        report_path,
        {
            "report_type": "safe-margin-mvp-development-report",
            "strategy": STRATEGY_LABEL,
            "input_path": _repo_relative(paths["input"]),
            "outcomes_path": _repo_relative(paths["outcomes"]),
            "artifact_path": _repo_relative(artifact_path),
            "tier_plans": {
                tier: {
                    "target_ratio": plans[tier].target_ratio,
                    "predicted_budget_ratio": plans[tier].predicted_budget_ratio,
                    "model_counts": dict(plans[tier].model_counts),
                    "stages": [
                        {
                            "step": stage.step,
                            "considered": stage.considered,
                            "eligible": stage.eligible,
                            "promoted": stage.promoted,
                            "groups_considered": stage.groups_considered,
                            "groups_promoted": stage.groups_promoted,
                            "budget_ratio": stage.budget,
                            "spent_ratio": stage.spent_ratio,
                        }
                        for stage in plans[tier].stages
                    ],
                }
                for tier in TIERS
            },
            "public_dev_baselines": {
                name: {tier: list(values[tier]) for tier in (*TIERS, "final")}
                for name, values in PUBLIC_DEV_BASELINES.items()
            },
            "score": report,
        },
    )
    if not quiet:
        _print_report(report, plans)
        print(f"제출 파일과 보고서: {output_directory}")
    return {"report": report, "plans": plans, "report_path": report_path}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_mvp",
        description="safe-margin MVP 라우터의 개발용 한 번 실행 도구",
    )
    parser.add_argument("--split", default="dev", choices=("train", "dev"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_mvp(
            split=args.split,
            input_path=args.input,
            outcomes_path=args.outcomes,
            artifact_path=args.artifact,
            policy_path=args.policy,
            output_directory=args.output_dir,
            quiet=args.quiet,
        )
    except (
        MvpRunError,
        OSError,
        ProtocolError,
        ScoringError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    failed: List[str] = [
        tier
        for tier in TIERS
        if not result["report"]["tiers"][tier]["budget_passed"]
    ]
    if failed:
        print(f"오류: 예산을 초과한 등급: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
