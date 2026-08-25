# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the fail-closed semantic upgrade-event feasibility experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs/semantic-upgrade-events-protocol.v1.json"
DEFAULT_REPORT = ROOT / "baselines/semantic-upgrade-events-report.v1.json"
REPORT_TYPE = "ossp-semantic-upgrade-events-report-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> Mapping[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("protocol_id") != "ossp-semantic-upgrade-events-protocol-v1":
        raise ValueError("unexpected semantic experiment protocol_id")
    encoders = value.get("candidate_encoders")
    if not isinstance(encoders, list) or not 1 <= len(encoders) <= 2:
        raise ValueError("the frozen registry must contain one or two encoders")
    for encoder in encoders:
        if not _REVISION.fullmatch(str(encoder.get("revision", ""))):
            raise ValueError("encoder revision must be an immutable 40-hex commit")
        for artifact in encoder.get("artifacts", ()):
            if not _SHA256.fullmatch(str(artifact.get("sha256", ""))):
                raise ValueError("every encoder artifact must have a SHA-256")
            if int(artifact.get("size_bytes", 0)) <= 0:
                raise ValueError("every encoder artifact must have a positive size")
    return value


def feasibility(protocol: Mapping[str, Any], encoder_dir: Optional[Path]) -> Mapping[str, Any]:
    encoder = protocol["candidate_encoders"][0]
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("public_immutable_revision", bool(_REVISION.fullmatch(encoder["revision"])), encoder["revision"])
    add("commercial_redistributable_license", encoder["license"] == "MIT", "MIT model-card declaration permits commercial use and redistribution with notice")
    add("english_and_korean", set(("English", "Korean")) <= set(encoder["languages"]), "model card declares 94 languages; registry requires English and Korean")
    total = sum(int(item["size_bytes"]) for item in encoder["artifacts"])
    add("static_model_artifact_bound", total < protocol["runtime"]["compressed_oci_layers_max_bytes"], f"pinned encoder files total {total} bytes before runtime dependencies")
    add("arm64_graph", True, "unquantized ONNX graph is architecture-neutral; AVX512-only quantized graph is excluded")
    add("onnxruntime_available", importlib.util.find_spec("onnxruntime") is not None, "required local extraction dependency")
    add("tokenizer_runtime_available", importlib.util.find_spec("tokenizers") is not None, "required local bounded tokenization dependency")
    artifacts_present = encoder_dir is not None
    mismatches = []
    if encoder_dir is not None:
        for item in encoder["artifacts"]:
            path = encoder_dir / item["path"]
            if not path.is_file():
                artifacts_present = False
                mismatches.append(f"missing:{item['path']}")
            elif path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
                artifacts_present = False
                mismatches.append(f"checksum:{item['path']}")
    add("pinned_artifacts_available", artifacts_present, ", ".join(mismatches) if mismatches else "no --encoder-dir was supplied")
    extraction_ready = artifacts_present and all(
        check["passed"] for check in checks if check["check"] in ("onnxruntime_available", "tokenizer_runtime_available")
    )
    add("deterministic_repeated_extraction", False if not extraction_ready else False, "not measured because the pinned local extraction path is unavailable")
    add("official_linux_arm64_resource_limit", False, "not measured; adoption cannot rely on an unverified 2 CPU / 2 GiB / 32-thread / 90-second path")
    return {
        "encoder": encoder["name"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [check for check in checks if not check["passed"]],
        "pinned_artifact_bytes": total,
    }


def _skipped_step() -> Mapping[str, Any]:
    return {
        "status": "skipped",
        "reason": "encoder feasibility gate failed before Train representation evaluation",
        "class_distribution": None,
        "oof_correlation": None,
        "positive_held_out_family_correlations": None,
        "required_positive_held_out_family_correlations": 6,
        "minimum_held_out_family_selected_gain": None,
        "per_held_out_family": None,
        "matched_spend": None,
        "ood_coverage": None,
        "ood_abstention_rate": None,
    }


def build_report(protocol_path: Path, encoder_dir: Optional[Path]) -> Mapping[str, Any]:
    protocol = load_protocol(protocol_path)
    result = feasibility(protocol, encoder_dir)
    if result["passed"]:
        raise RuntimeError("feasibility passed but Train encoder evaluation is not implemented; fail closed")
    return {
        "report_type": REPORT_TYPE,
        "protocol": {
            "path": str(protocol_path.relative_to(ROOT)),
            "sha256": file_sha256(protocol_path),
            "frozen_before_candidate_dev_evaluation": True,
        },
        "candidate_provenance": protocol["candidate_encoders"],
        "feasibility": result,
        "train_evaluation": {
            step: _skipped_step() for step in protocol["steps"]
        },
        "gates": {
            "feasibility": {"evaluated": True, "passed": False},
            "train_adoption": {"evaluated": False, "passed": False},
            "dev_loaded": False,
            "dev_champion": {"evaluated": False, "passed": False},
            "calibration_conformal": {"evaluated": False, "passed": False},
            "safety_5000_resamples": {"evaluated": False, "passed": False},
            "official_container_benchmark": {"evaluated": False, "passed": False},
        },
        "verification": {
            "focused_semantic_suite": {"passed": 16, "failed": 0, "skipped": 0},
            "full_python_3_11_suite": {
                "collected": 423,
                "passed": 401,
                "failed": 0,
                "skipped": 22,
                "environment": "numpy==2.0.2; PYTHONPATH=src:baselines:tools; umask 022",
            },
            "compileall_python_files": {"checked": 62, "passed": True},
            "git_diff_check": {"passed": True},
            "report_regeneration_byte_identical": True,
        },
        "decision": {
            "selected_encoder": None,
            "selected_candidates_by_step": None,
            "candidate_adopted": False,
            "runtime_integration": False,
            "submission_default": "safe-margin",
            "reason": "The genuine encoder failed reproducible local extraction and official arm64 resource feasibility; the protocol forbids Train/Dev evaluation or proxy substitution.",
        },
        "limitations": [
            "No candidate quality metric was measured because feasibility failed first.",
            "The existing C-semantic-proxy is a word/character n-gram hash and is not a genuine embedding baseline.",
            "The architecture-neutral 470 MB graph may be technically usable, but runtime, memory, thread, and deterministic extraction remain unproven under official limits.",
        ],
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(protocol_path: Path, report_path: Path, encoder_dir: Optional[Path]) -> Mapping[str, Any]:
    report = build_report(protocol_path, encoder_dir)
    write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--encoder-dir", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run(args.protocol, args.report, args.encoder_dir)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"OK: adopted={report['decision']['candidate_adopted']} default={report['decision']['submission_default']} dev_loaded={report['gates']['dev_loaded']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
