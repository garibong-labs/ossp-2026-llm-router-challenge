# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0
"""Measure and reproduce the frozen semantic upgrade-event experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools", ROOT / "baselines"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import numpy as np  # noqa: E402
import representation_features as structural  # noqa: E402
import risk_validation  # noqa: E402
import semantic_upgrade_events as events  # noqa: E402
import train_risk_calibrated as trainer  # noqa: E402
from ossp_router.protocol import load_bundled_policy, load_input, load_outcomes  # noqa: E402

DEFAULT_PROTOCOL = ROOT / "configs/semantic-upgrade-events-protocol.v1.json"
DEFAULT_REPORT = ROOT / "baselines/semantic-upgrade-events-report.v1.json"
DEFAULT_EVIDENCE = ROOT / "baselines/semantic-upgrade-events-evidence.v1.json"
DEFAULT_TRAIN_INPUT = ROOT / "data/materialized/train/inputs.json"
DEFAULT_TRAIN_OUTCOMES = ROOT / "data/train/outcomes.json"
REPORT_TYPE = "ossp-semantic-upgrade-events-report-v1"
EVIDENCE_TYPE = "ossp-semantic-upgrade-events-evidence-v1"
PINNED_MODEL_CARD_SHA256 = "ea680357ec21065558db494afe9092e129fe5605a9dca80b3d5b1c144c2d1552"
REPRESENTATIONS = (
    "semantic",
    "semantic+expanded-structural-b",
    "semantic+expanded-structural-b+ood-retrieval-confidence",
)
STEP_TIERS = {events.STEPS[0]: ("fast", "balanced"), events.STEPS[1]: ("premium",)}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_bytes())
    if value.get("protocol_id") != "ossp-semantic-upgrade-events-protocol-v1":
        raise ValueError("unexpected semantic experiment protocol_id")
    encoders = value.get("candidate_encoders")
    if not isinstance(encoders, list) or not 1 <= len(encoders) <= 2:
        raise ValueError("registry must contain one or two encoders")
    for encoder in encoders:
        revision = str(encoder.get("revision", ""))
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError("encoder revision must be immutable 40-hex")
        for artifact in encoder.get("artifacts", ()):
            digest = str(artifact.get("sha256", ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("every artifact needs SHA-256")
            if int(artifact.get("size_bytes", 0)) <= 0:
                raise ValueError("every artifact needs a positive size")
    return value


def verify_artifacts(protocol: Mapping[str, Any], encoder_dir: Path) -> Mapping[str, Any]:
    encoder = protocol["candidate_encoders"][0]
    files = []
    for expected in encoder["artifacts"]:
        path = encoder_dir / expected["path"]
        size = path.stat().st_size if path.is_file() else None
        digest = file_sha256(path) if size == expected["size_bytes"] else None
        files.append({
            "path": expected["path"], "expected_size_bytes": expected["size_bytes"],
            "observed_size_bytes": size, "expected_sha256": expected["sha256"],
            "observed_sha256": digest,
            "passed": size == expected["size_bytes"] and digest == expected["sha256"],
        })
    return {
        "revision": encoder["revision"], "files": files,
        "artifact_bytes": sum(row["size_bytes"] for row in encoder["artifacts"]),
        "passed": all(row["passed"] for row in files),
    }


def provision_artifacts(
    protocol: Mapping[str, Any], encoder_dir: Path, *, opener: Any = urllib.request.urlopen
) -> Mapping[str, Any]:
    """Download only registry-listed immutable files and verify before use."""

    encoder = protocol["candidate_encoders"][0]
    template = encoder["build_time_download"]["url_template"]
    for expected in encoder["artifacts"]:
        destination = encoder_dir / expected["path"]
        if (
            destination.is_file()
            and destination.stat().st_size == expected["size_bytes"]
            and file_sha256(destination) == expected["sha256"]
        ):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.download-{os.getpid()}")
        try:
            with opener(template.format(artifact_path=expected["path"]), timeout=60) as response, temporary.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if temporary.stat().st_size != expected["size_bytes"] or file_sha256(temporary) != expected["sha256"]:
                raise ValueError(f"downloaded artifact verification failed: {expected['path']}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return verify_artifacts(protocol, encoder_dir)


def dependency_evidence() -> Mapping[str, Any]:
    packages = []
    for name in ("numpy", "onnxruntime", "tokenizers", "psutil"):
        distribution = importlib.metadata.distribution(name)
        installed = sum(
            path.stat().st_size for item in distribution.files or ()
            if (path := Path(distribution.locate_file(item))).is_file()
        )
        packages.append({"name": name, "version": distribution.version, "installed_bytes": installed})
    return {
        "python": platform.python_version(), "packages": packages,
        "required_installed_bytes": sum(row["installed_bytes"] for row in packages),
        "network_required_at_evaluation_runtime": False,
    }


def measure_extraction(encoder_dir: Path, episodes: Sequence[Any]) -> Tuple[Any, Mapping[str, Any]]:
    import psutil
    process = psutil.Process()
    def child_count() -> int:
        try:
            return len(process.children(recursive=True))
        except (OSError, psutil.Error):
            return 0
    before_threads = process.num_threads()
    before_children = child_count()
    encoder = events.SemanticEncoder(encoder_dir, threads=2)
    started = time.perf_counter(); first = encoder.encode(episodes, batch_size=1); first_seconds = time.perf_counter() - started
    first_rss = process.memory_info().rss
    middle_threads = process.num_threads()
    started = time.perf_counter(); second = encoder.encode(episodes, batch_size=1); second_seconds = time.perf_counter() - started
    after_threads = process.num_threads()
    after_children = child_count()
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    peak = max(peak, first_rss)
    norm_error = float(np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0)))
    maximum_processes = max(before_threads, middle_threads, after_threads) + after_children
    passed = (
        first_seconds <= 90 and second_seconds <= 90 and peak <= 2_147_483_648
        and maximum_processes <= 32 and first.tobytes() == second.tobytes()
        and np.isfinite(first).all() and norm_error <= 1e-5
    )
    return first, {
        "host": {"system": platform.system(), "machine": platform.machine(), "release": platform.release()},
        "official_linux_arm64": platform.system() == "Linux" and platform.machine() in ("arm64", "aarch64"),
        "official_path_status": "measured" if platform.system() == "Linux" and platform.machine() in ("arm64", "aarch64") else "unavailable-after-command-discovery",
        "container_vm_attempts": {name: shutil.which(name) for name in ("docker", "podman", "colima", "limactl")},
        "control": {"cpu_count": 2, "onnx_intra_op_threads": 2, "onnx_inter_op_threads": 1, "execution_mode": "sequential", "batch_size": 1, "runtime_network": False},
        "rows": len(episodes), "dimensions": int(first.shape[1]),
        "first_seconds": first_seconds, "second_seconds": second_seconds,
        "rows_per_second_second_pass": len(episodes) / second_seconds,
        "byte_identical": first.tobytes() == second.tobytes(), "maximum_norm_error": norm_error,
        "rss_after_first_bytes": first_rss, "peak_rss_bytes": peak,
        "threads_before": before_threads, "threads_after_first": middle_threads,
        "threads_after_second": after_threads, "child_pids_before": before_children,
        "child_pids_after": after_children, "maximum_pid_thread_observation": maximum_processes,
        "limits": {"cpu_count": 2, "memory_max_bytes": 2_147_483_648, "pid_thread_limit": 32, "seconds_per_tier": 90},
        "constraint_passed": passed,
    }


def _matrices(inputs: Any, embeddings: Any) -> Mapping[str, Any]:
    shape = np.asarray([structural.expanded_structural_vector(row) for row in inputs.episodes])
    combined = np.column_stack((embeddings, shape))
    return {REPRESENTATIONS[0]: embeddings, REPRESENTATIONS[1]: combined, REPRESENTATIONS[2]: combined}


def _projection(train: Any, test: Any, name: str) -> Tuple[Any, Any, Mapping[str, Any]]:
    selected = np.argsort(-np.var(train[:, :384], axis=0), kind="stable")[:64]
    train_result, test_result = train[:, selected], test[:, selected]
    if name != REPRESENTATIONS[0]:
        train_result = np.column_stack((train_result, train[:, 384:]))
        test_result = np.column_stack((test_result, test[:, 384:]))
    record = {"fit_scope": "supplied-family-fold-training-only", "method": "variance-ranked", "semantic_dimensions": [int(v) for v in selected], "output_dimensions": int(train_result.shape[1])}
    return train_result, test_result, record


def _cost_prediction(train_x: Any, test_x: Any, increments: Any) -> Any:
    model = events._weighted_ridge(train_x, np.log(np.maximum(increments, 1e-12)), 100.0)
    train_pred = np.exp(events._ridge_predict(model, train_x).reshape(-1))
    factor = float(np.quantile(np.maximum(increments, 0) / np.maximum(train_pred, 1e-12), .95, method="inverted_cdf"))
    return np.exp(events._ridge_predict(model, test_x).reshape(-1)) * max(1.0, factor)


def _ood_mask(train: Any, test: Any, labels: Any, probabilities: Any, committee: Any, config: Mapping[str, Any], cache: Optional[dict] = None, cache_key: Any = None) -> Tuple[Any, Mapping[str, Any]]:
    support_key = (cache_key, "support-matrices")
    stored = cache.get(support_key) if cache is not None else None
    if stored is None:
        train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-12)
        similarities = test @ train.T
        self_similarity = train @ train.T
        np.fill_diagonal(self_similarity, -math.inf)
        stored = (similarities, self_similarity.max(axis=1))
        if cache is not None: cache[support_key] = stored
    similarities, leave_self_nearest = stored
    k = int(config["nearest_neighbors"])
    indices = np.argsort(-similarities, axis=1, kind="stable")[:, :k]
    nearest = np.take_along_axis(similarities, indices, axis=1)[:, 0]
    agreement = np.asarray([max(Counter(int(v) for v in labels[row]).values()) / k for row in indices])
    threshold = float(np.quantile(leave_self_nearest, float(config["support_similarity_quantile"]), method="inverted_cdf"))
    confidence = float(config["committee_confidence_minimum"])
    passed = ((nearest >= threshold) & (agreement >= float(config["local_label_agreement_minimum"])) & (probabilities.max(axis=1) >= confidence) & (committee >= confidence))
    return passed, {"support_similarity_threshold": threshold, "coverage": float(passed.mean()), "abstention_rate": float(1 - passed.mean())}


def _options(protocol: Mapping[str, Any], name: str) -> Tuple[Mapping[str, Any], ...]:
    grid = protocol["hyperparameter_grid"]
    keys = ["classifier_l2", "magnitude_ridge_l2"]
    if name == REPRESENTATIONS[2]:
        keys += ["nearest_neighbors", "support_similarity_quantile", "local_label_agreement_minimum", "committee_confidence_minimum"]
    return tuple(dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys)))


def _fold(matrix: Any, embedding: Any, gains: Any, increments: Any, train: Any, test: Any, name: str, config: Mapping[str, Any], cache: Optional[dict] = None, cache_key: Any = None) -> Mapping[str, Any]:
    model_name = "semantic" if name == REPRESENTATIONS[0] else "semantic+structural"
    base_key = (cache_key, model_name, config["classifier_l2"], config["magnitude_ridge_l2"])
    base = cache.get(base_key) if cache is not None else None
    if base is None:
        train_x, test_x, projection = _projection(matrix[train], matrix[test], name)
        model = events.fit_event_utility_model(train_x, gains[train], classifier_l2=float(config["classifier_l2"]), magnitude_ridge_l2=float(config["magnitude_ridge_l2"]))
        prediction = events.predict_event_utility(model, test_x)
        base = {"utility": prediction["expected_utility"], "increment": _cost_prediction(train_x, test_x, increments[train]), "projection": projection, "probabilities": prediction["probabilities"], "labels": model["labels"], "train_x": train_x, "test_x": test_x}
        if cache is not None: cache[base_key] = base
    permitted = np.ones(len(test), dtype=bool); ood = {"coverage": 1.0, "abstention_rate": 0.0}
    if name == REPRESENTATIONS[2]:
        vote_key = (cache_key, model_name, "committee", config["magnitude_ridge_l2"])
        votes = cache.get(vote_key) if cache is not None else None
        if votes is None:
            votes = []
            for alpha in (0.1, 1.0, 10.0):
                member = events.fit_event_utility_model(base["train_x"], gains[train], classifier_l2=alpha, magnitude_ridge_l2=float(config["magnitude_ridge_l2"]))
                votes.append(events.predict_event_utility(member, base["test_x"])["expected_utility"] > 0)
            votes = np.mean(votes, axis=0)
            if cache is not None: cache[vote_key] = votes
        permitted, ood = _ood_mask(embedding[train], embedding[test], base["labels"], base["probabilities"], votes, config, cache, cache_key)
    return {"utility": base["utility"], "increment": base["increment"], "permitted": permitted, "projection": base["projection"], "ood": ood}


def _matched(prediction: Mapping[str, Any], gains: Any, increments: Any, costs: Any, goal: float, step: int) -> Mapping[str, Any]:
    light = float(costs[:, 0].sum())
    budget = (goal - 1) * light if step == 0 else max(0.0, goal * light - float(costs[:, 1].sum()))
    order = np.argsort(-(prediction["utility"] / np.maximum(prediction["increment"], 1e-12)), kind="stable")
    selected, used = [], 0.0
    for raw in order:
        index = int(raw); cost = max(0.0, float(prediction["increment"][index]))
        if prediction["permitted"][index] and prediction["utility"][index] > 0 and used + cost <= budget:
            selected.append(index); used += cost
    realized_cost = math.fsum(float(increments[index]) for index in selected)
    return {"selected_count": len(selected), "realized_incremental_gain": math.fsum(float(gains[index]) for index in selected) / len(gains), "realized_incremental_cost_over_light": realized_cost / max(light, 1e-12), "predicted_budget_used": used, "matched_incremental_budget": budget}


def _score(prediction: Mapping[str, Any], gains: Any, increments: Any, costs: Any, tiers: Sequence[str], goals: Mapping[str, float], step: int) -> Tuple[float, Mapping[str, Any]]:
    matched = {tier: _matched(prediction, gains, increments, costs, goals[tier], step) for tier in tiers}
    return sum(row["realized_incremental_gain"] for row in matched.values()), matched


def nested_train_evaluation(protocol: Mapping[str, Any], inputs: Any, embedding: Any, scores: Any, costs: Any, families: Sequence[str]) -> Mapping[str, Any]:
    matrices = _matrices(inputs, embedding)
    gains = np.column_stack((scores[:, 1] - scores[:, 0], scores[:, 2] - scores[:, 1]))
    increments = np.column_stack((costs[:, 1] - costs[:, 0], costs[:, 2] - costs[:, 1]))
    goals = protocol["matched_spend"]["spend_goals_over_all_light"]
    result = {}
    for step_index, step_name in enumerate(events.STEPS):
        utility = np.zeros(len(families)); predicted_cost = np.zeros(len(families)); permitted = np.zeros(len(families), dtype=bool)
        outer_records = []
        for outer in events.nested_family_splits(families):
            outer_train, outer_test = np.asarray(outer["train"]), np.asarray(outer["test"])
            position = {int(row): index for index, row in enumerate(outer_train)}
            best = None; objectives = {}; fold_cache = {}
            for name in REPRESENTATIONS:
                for option_index, config in enumerate(_options(protocol, name)):
                    inner = {"utility": np.zeros(len(outer_train)), "increment": np.zeros(len(outer_train)), "permitted": np.zeros(len(outer_train), dtype=bool)}
                    for fold in outer["inner"]:
                        train, test = np.asarray(fold["train"]), np.asarray(fold["test"])
                        prediction = _fold(matrices[name], embedding, gains[:, step_index], increments[:, step_index], train, test, name, config, fold_cache, fold["held_out_family"])
                        locations = np.asarray([position[int(row)] for row in test])
                        for key in inner: inner[key][locations] = prediction[key]
                    objective, _ = _score(inner, gains[outer_train, step_index], increments[outer_train, step_index], costs[outer_train], STEP_TIERS[step_name], goals, step_index)
                    key = f"{name}#{option_index}"; objectives[key] = objective
                    rank = (objective, events.correlation(inner["utility"], gains[outer_train, step_index]), name, -option_index)
                    if best is None or rank > best[0]: best = (rank, name, config)
            assert best is not None
            _, name, config = best
            prediction = _fold(matrices[name], embedding, gains[:, step_index], increments[:, step_index], outer_train, outer_test, name, config)
            utility[outer_test], predicted_cost[outer_test], permitted[outer_test] = prediction["utility"], prediction["increment"], prediction["permitted"]
            outer_records.append({
                "held_out_family": outer["held_out_family"],
                "candidate": name,
                "hyperparameters": config,
                "projection": prediction["projection"],
                "fit_objects": {
                    "scope": "outer-family-training-portion-only",
                    "scaling": "per-feature mean and standard deviation",
                    "class_imbalance": "inverse-frequency weights normalized to mean one",
                    "event_calibration": {"method": "softmax", "temperature": 1.0},
                    "conditional_magnitudes": ["win ridge", "absolute-loss ridge"],
                    "incremental_cost": "ridge log-cost with training-residual p95 conservative multiplier",
                },
                "ood": prediction["ood"],
                "inner_candidate_hyperparameter_objectives": objectives,
            })
        prediction = {"utility": utility, "increment": predicted_cost, "permitted": permitted}
        _, matched = _score(prediction, gains[:, step_index], increments[:, step_index], costs, STEP_TIERS[step_name], goals, step_index)
        per_family = {}
        for family in sorted(set(families)):
            mask = np.asarray([value == family for value in families])
            _, family_matched = _score({key: value[mask] for key, value in prediction.items()}, gains[mask, step_index], increments[mask, step_index], costs[mask], STEP_TIERS[step_name], goals, step_index)
            per_family[family] = {"rows": int(mask.sum()), "oof_correlation": events.correlation(utility[mask], gains[mask, step_index]), "matched_spend": family_matched}
        labels = events.event_labels(scores[:, step_index], scores[:, step_index + 1])
        result[step_name] = {
            "status": "completed", "class_distribution": {events.EVENTS[i]: int((labels == i).sum()) for i in range(3)},
            "oof_correlation": events.correlation(utility, gains[:, step_index]),
            "positive_held_out_family_correlations": sum(row["oof_correlation"] > 0 for row in per_family.values()),
            "required_positive_held_out_family_correlations": 6,
            "minimum_held_out_family_selected_gain": min(metric["realized_incremental_gain"] for row in per_family.values() for metric in row["matched_spend"].values()),
            "matched_spend": matched, "ood_coverage": float(permitted.mean()), "ood_abstention_rate": float(1 - permitted.mean()),
            "per_held_out_family": per_family, "outer_selections": outer_records,
            "candidate_selection_counts": dict(sorted(Counter(row["candidate"] for row in outer_records).items())),
        }
    return result


def train_gate(protocol: Mapping[str, Any], evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    threshold = protocol["adoption_thresholds"]
    reference = json.loads((ROOT / "baselines/representation-audit-report.v1.json").read_text())["train_representations"]["A-current-dense-wordhash"]["metrics"]
    checks = []
    for step in events.STEPS:
        row = evaluation[step]
        for label, observed, required in (
            ("oof_correlation", row["oof_correlation"], threshold["oof_correlation_each_step"]),
            ("positive_families", row["positive_held_out_family_correlations"], threshold["positive_held_out_family_correlations_each_step"]),
            ("family_gain_floor", row["minimum_held_out_family_selected_gain"], threshold["minimum_held_out_family_selected_gain"]),
        ): checks.append({"check": f"{step}.{label}", "observed": observed, "required": required, "passed": observed >= required})
        for tier in STEP_TIERS[step]:
            observed = row["matched_spend"][tier]["realized_incremental_gain"]
            required = reference[step]["selected_set"][tier]["realized_incremental_gain"] + threshold["selected_gain_margin_over_current_a"]
            checks.append({"check": f"{step}.{tier}.beats_current_a", "observed": observed, "required": required, "passed": observed >= required})
    return {"evaluated": True, "passed": all(row["passed"] for row in checks), "checks": checks, "failed_checks": [row for row in checks if not row["passed"]]}


def run_measurement(protocol_path: Path, encoder_dir: Path, train_input: Path, train_outcomes: Path) -> Mapping[str, Any]:
    protocol = load_protocol(protocol_path); artifacts = verify_artifacts(protocol, encoder_dir)
    if not artifacts["passed"]: raise ValueError("pinned artifact verification failed")
    inputs, outcomes, policy = load_input(train_input), load_outcomes(train_outcomes), load_bundled_policy()
    _, scores, costs = trainer._training_tables(inputs, outcomes, policy, 16)
    embedding, benchmark = measure_extraction(encoder_dir, inputs.episodes)
    if not benchmark["constraint_passed"]: raise RuntimeError("measured extraction feasibility failed")
    evaluation = nested_train_evaluation(protocol, inputs, embedding, scores, costs, risk_validation.reconstruct_families("train", inputs))
    encoder = protocol["candidate_encoders"][0]
    registry = {"revision": encoder["revision"], "license": encoder["license"], "languages": encoder["languages"], "model_card": encoder["model_card"], "license_evidence": encoder["license_evidence"], "downloaded_model_card_sha256": PINNED_MODEL_CARD_SHA256, "query_prefix": events.QUERY_PREFIX, "pooling": "attention-mask mean pooling followed by L2 normalization"}
    official_environment = benchmark["official_linux_arm64"]
    native_apple_arm64 = benchmark["host"]["system"] == "Darwin" and benchmark["host"]["machine"] == "arm64"
    official_feasibility = {
        "required_environment": protocol["runtime"]["official_architecture"],
        "evaluated": official_environment,
        "passed": official_environment and benchmark["constraint_passed"],
        "status": "measured" if official_environment else benchmark["official_path_status"],
    }
    return {"evidence_type": EVIDENCE_TYPE, "protocol_sha256": file_sha256(protocol_path), "train_input_sha256": file_sha256(train_input), "train_outcomes_sha256": file_sha256(train_outcomes), "registry_evidence": registry, "artifacts": artifacts, "dependencies": dependency_evidence(), "native_apple_arm64_preflight": {"required_environment": "darwin/arm64", "evaluated": native_apple_arm64, "passed": native_apple_arm64 and benchmark["constraint_passed"], "extraction_benchmark": benchmark}, "official_linux_arm64_feasibility": official_feasibility, "train_evaluation": evaluation, "train_gate": train_gate(protocol, evaluation)}


def build_report(protocol_path: Path, evidence_path: Path = DEFAULT_EVIDENCE) -> Mapping[str, Any]:
    protocol = load_protocol(protocol_path); evidence = json.loads(evidence_path.read_text())
    if evidence.get("evidence_type") != EVIDENCE_TYPE or evidence.get("protocol_sha256") != file_sha256(protocol_path): raise ValueError("evidence does not match protocol")
    native_preflight = evidence["native_apple_arm64_preflight"]
    official_feasibility = evidence["official_linux_arm64_feasibility"]
    installed_and_model = evidence["artifacts"]["artifact_bytes"] + evidence["dependencies"]["required_installed_bytes"]
    static_size_passed = installed_and_model < protocol["runtime"]["compressed_oci_layers_max_bytes"]
    feasible = (
        evidence["artifacts"]["passed"]
        and official_feasibility["evaluated"]
        and official_feasibility["passed"]
        and static_size_passed
    )
    return {
        "report_type": REPORT_TYPE, "protocol": {"path": str(protocol_path.relative_to(ROOT)), "sha256": file_sha256(protocol_path), "frozen_before_candidate_dev_evaluation": True},
        "candidate_provenance": protocol["candidate_encoders"], "registry_evidence": evidence["registry_evidence"], "artifact_verification": evidence["artifacts"], "runtime_dependencies": evidence["dependencies"],
        "feasibility": {"passed": feasible, "encoder": protocol["candidate_encoders"][0]["name"], "native_apple_arm64_preflight": native_preflight, "official_linux_arm64": official_feasibility, "model_plus_required_runtime_bytes": installed_and_model, "compressed_oci_layer_bound_bytes": protocol["runtime"]["compressed_oci_layers_max_bytes"], "static_size_bound_passed": static_size_passed},
        "train_protocol_execution": {"selection": "independent per step inside each outer fold", "outer": protocol["splits"]["outer"], "inner": protocol["splits"]["inner"], "candidate_feature_combinations": protocol["candidate_feature_combinations"], "fit_scope": "each inner/outer training portion only", "fold_local_objects": ["inverse-frequency weights", "scaling", "variance projection", "event heads", "conditional magnitude heads", "cost head and conservative residual multiplier", "retrieval/OOD thresholds", "calibration object", "hyperparameters"]},
        "train_evaluation": evidence["train_evaluation"],
        "gates": {"feasibility": {"required_environment": official_feasibility["required_environment"], "evaluated": official_feasibility["evaluated"], "passed": feasible, "status": official_feasibility["status"]}, "train_adoption": evidence["train_gate"], "dev_loaded": False, "dev_champion": {"evaluated": False, "passed": False}, "calibration_conformal": {"evaluated": False, "passed": False}, "safety_5000_resamples": {"evaluated": False, "passed": False}, "official_container_benchmark": {"evaluated": False, "passed": False}},
        "verification": evidence.get("verification", {}),
        "decision": {"selected_encoder": protocol["candidate_encoders"][0]["name"], "selected_candidates_by_step": {step: max(row["candidate_selection_counts"], key=lambda name: (row["candidate_selection_counts"][name], name)) for step, row in evidence["train_evaluation"].items()}, "candidate_adopted": False, "runtime_integration": False, "submission_default": "safe-margin", "reason": "Completed nested Train quality gate failed; Dev, safety, and official container gates remained closed."},
        "limitations": ["Official linux/arm64 was unavailable after Docker, Podman, Colima, and Lima discovery; resource evidence is native Apple arm64 under frozen ONNX thread controls.", "Nine public families provide limited family-level power.", "The completed Train failure prevents Dev outcome loading and runtime integration."],
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try: temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"); temporary.chmod(0o644); os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def run(protocol_path: Path, report_path: Path, evidence_path: Path = DEFAULT_EVIDENCE) -> Mapping[str, Any]:
    report = build_report(protocol_path, evidence_path); write_json_atomic(report_path, report); return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL); parser.add_argument("--report", type=Path, default=DEFAULT_REPORT); parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE); parser.add_argument("--measure", action="store_true"); parser.add_argument("--provision", action="store_true"); parser.add_argument("--encoder-dir", type=Path); parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT); parser.add_argument("--train-outcomes", type=Path, default=DEFAULT_TRAIN_OUTCOMES)
    args = parser.parse_args(argv)
    try:
        if args.measure:
            if args.encoder_dir is None: raise ValueError("--measure requires --encoder-dir")
            if args.provision: provision_artifacts(load_protocol(args.protocol), args.encoder_dir)
            write_json_atomic(args.evidence, run_measurement(args.protocol, args.encoder_dir, args.train_input, args.train_outcomes))
        report = run(args.protocol, args.report, args.evidence)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc: print(f"error: {exc}", file=sys.stderr); return 2
    print(f"OK: adopted={report['decision']['candidate_adopted']} default={report['decision']['submission_default']} dev_loaded={report['gates']['dev_loaded']}"); return 0


if __name__ == "__main__": raise SystemExit(main())
