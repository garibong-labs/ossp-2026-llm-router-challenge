# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Train-only primitives for the frozen semantic upgrade-event experiment.

This is deliberately separate from the submitted runtime.  It contains no
encoder substitute: a semantic candidate is usable only when the pinned
pretrained encoder has passed the feasibility checks in the protocol.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ossp_router.protocol import Episode


MAX_FIELD_CHARACTERS = 32_768
EVENTS = ("loss", "tie", "win")
STEPS = ("ax31-light->ax31", "ax31->axk1-think")
QUERY_PREFIX = "query: "
MAX_TOKENS = 512


def bounded_role_content(episode: Episode) -> Tuple[Tuple[str, str], ...]:
    """Return only bounded runtime-authorized role/content fields."""

    if episode.prompt is not None:
        return (("user", episode.prompt[:MAX_FIELD_CHARACTERS]),)
    assert episode.messages is not None
    return tuple(
        (message.role, message.content[:MAX_FIELD_CHARACTERS])
        for message in episode.messages
    )


def semantic_text(episode: Episode) -> str:
    """Serialize only bounded role/content fields using E5's query prefix."""

    fields = bounded_role_content(episode)
    return QUERY_PREFIX + "\n".join(f"[{role}]\n{content}" for role, content in fields)


class SemanticEncoder:
    """Offline, bounded multilingual-E5 ONNX feature extractor.

    Imports are intentionally lazy so the experiment's pure Train primitives
    and tests do not make ONNX Runtime a submitted-runtime dependency.
    """

    def __init__(self, encoder_dir: Path, *, threads: int = 2) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise RuntimeError("onnxruntime and tokenizers are required") from exc
        root = Path(encoder_dir) / "onnx"
        tokenizer_path = root / "tokenizer.json"
        model_path = root / "model.onnx"
        if not tokenizer_path.is_file() or not model_path.is_file():
            raise ValueError("verified encoder directory is incomplete")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=MAX_TOKENS)
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._tokenizer = tokenizer
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.threads = threads

    def encode(self, episodes: Sequence[Episode], *, batch_size: int = 16) -> Any:
        """Mean-pool and L2-normalize deterministic float32 embeddings."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        rows = []
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start : start + batch_size]
            encoded = self._tokenizer.encode_batch([semantic_text(item) for item in batch])
            feeds = {
                "input_ids": np.asarray([item.ids for item in encoded], dtype=np.int64),
                "attention_mask": np.asarray(
                    [item.attention_mask for item in encoded], dtype=np.int64
                ),
                "token_type_ids": np.asarray(
                    [item.type_ids for item in encoded], dtype=np.int64
                ),
            }
            hidden = self._session.run(["last_hidden_state"], feeds)[0]
            mask = feeds["attention_mask"].astype(np.float32)[..., None]
            pooled = (hidden * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
            pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
            rows.append(pooled.astype(np.float32, copy=False))
        return np.vstack(rows) if rows else np.empty((0, 384), dtype=np.float32)


def extraction_probe(
    encoder: SemanticEncoder,
    episodes: Sequence[Episode],
    *,
    batch_size: int = 16,
) -> Mapping[str, Any]:
    """Measure two foreground passes and require byte-identical output."""

    started = time.perf_counter()
    first = encoder.encode(episodes, batch_size=batch_size)
    first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    second = encoder.encode(episodes, batch_size=batch_size)
    second_seconds = time.perf_counter() - started
    return {
        "rows": len(episodes),
        "dimensions": int(first.shape[1]) if len(first) else 384,
        "first_seconds": first_seconds,
        "second_seconds": second_seconds,
        "byte_identical": first.tobytes() == second.tobytes(),
        "finite": bool(np.isfinite(first).all()),
        "maximum_norm_error": (
            float(np.max(np.abs(np.linalg.norm(first, axis=1) - 1.0)))
            if len(first)
            else 0.0
        ),
    }


def event_labels(base_scores: Any, upgraded_scores: Any) -> Any:
    """Encode loss/tie/win as 0/1/2 without collapsing signed gain."""

    base = np.asarray(base_scores, dtype=np.float64)
    upgraded = np.asarray(upgraded_scores, dtype=np.float64)
    if base.shape != upgraded.shape or not np.isfinite(base).all() or not np.isfinite(upgraded).all():
        raise ValueError("score arrays must have the same finite shape")
    return np.where(upgraded > base, 2, np.where(upgraded < base, 0, 1)).astype(np.int64)


def normalized_inverse_frequency_weights(labels: Any) -> Any:
    """Return predeclared Train-fold-only inverse-frequency weights."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0) | (values > 2)):
        raise ValueError("labels must be a non-empty vector containing only 0, 1, 2")
    counts = Counter(int(value) for value in values)
    raw = np.asarray([1.0 / counts[int(value)] for value in values])
    return raw / raw.mean()


def conditional_magnitudes(signed_gains: Any, labels: Any) -> Tuple[float, float]:
    """Estimate positive win gain and absolute loss magnitude separately."""

    gains = np.asarray(signed_gains, dtype=np.float64)
    events = np.asarray(labels, dtype=np.int64)
    if gains.shape != events.shape or gains.ndim != 1 or not np.isfinite(gains).all():
        raise ValueError("gains and labels must be same-shaped finite vectors")
    wins = gains[events == 2]
    losses = -gains[events == 0]
    return (
        float(wins.mean()) if len(wins) else 0.0,
        float(losses.mean()) if len(losses) else 0.0,
    )


def expected_signed_utility(
    probabilities: Any, win_magnitude: Any, loss_magnitude: Any
) -> Any:
    """Compute P(win)*gain - P(loss)*loss with explicit loss penalty."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite N x 3 matrix")
    if np.any(values < 0.0) or np.any(np.abs(values.sum(axis=1) - 1.0) > 1e-9):
        raise ValueError("event probabilities must be nonnegative and sum to one")
    wins = np.asarray(win_magnitude, dtype=np.float64)
    losses = np.asarray(loss_magnitude, dtype=np.float64)
    if np.any(wins < 0.0) or np.any(losses < 0.0):
        raise ValueError("conditional magnitudes must be nonnegative")
    return values[:, 2] * wins - values[:, 0] * losses


def nested_family_splits(families: Sequence[str]) -> Tuple[Mapping[str, Any], ...]:
    """Build LOFO outer and inner indices with no family overlap."""

    names = tuple(sorted(set(families)))
    if len(names) < 3:
        raise ValueError("nested group validation requires at least three families")
    result = []
    for held_out in names:
        outer_train = tuple(i for i, family in enumerate(families) if family != held_out)
        outer_test = tuple(i for i, family in enumerate(families) if family == held_out)
        inner = []
        for inner_held_out in names:
            if inner_held_out == held_out:
                continue
            inner_train = tuple(i for i in outer_train if families[i] != inner_held_out)
            inner_test = tuple(i for i in outer_train if families[i] == inner_held_out)
            inner.append({"held_out_family": inner_held_out, "train": inner_train, "test": inner_test})
        result.append({"held_out_family": held_out, "train": outer_train, "test": outer_test, "inner": tuple(inner)})
    return tuple(result)


def select_step_candidates(
    objectives: Mapping[str, Mapping[str, float]]
) -> Mapping[str, str]:
    """Select each step independently; lexical name breaks exact ties."""

    selected = {}
    for step in STEPS:
        candidates = objectives.get(step, {})
        if not candidates:
            raise ValueError(f"missing Train-only objectives for {step}")
        selected[step] = max(candidates, key=lambda name: (candidates[name], name))
    return selected


def local_support(
    query: Any,
    training_embeddings: Any,
    training_labels: Any,
    *,
    neighbors: int,
    minimum_similarity: float,
    minimum_label_agreement: float,
    classifier_confidence: float,
    minimum_classifier_confidence: float,
    committee_agreement: float,
    minimum_committee_agreement: float,
) -> Mapping[str, Any]:
    """Evaluate conservative cosine/local-agreement/committee abstention."""

    point = np.asarray(query, dtype=np.float64)
    train = np.asarray(training_embeddings, dtype=np.float64)
    labels = np.asarray(training_labels, dtype=np.int64)
    if train.ndim != 2 or point.shape != (train.shape[1],) or labels.shape != (len(train),):
        raise ValueError("nearest-neighbor inputs have incompatible shapes")
    if neighbors < 1 or neighbors > len(train) or not np.isfinite(train).all() or not np.isfinite(point).all():
        raise ValueError("nearest-neighbor inputs must be finite and k must be in range")
    point_norm = float(np.linalg.norm(point))
    train_norms = np.linalg.norm(train, axis=1)
    similarities = train @ point / np.maximum(train_norms * max(point_norm, 1e-15), 1e-15)
    indices = np.argsort(-similarities, kind="stable")[:neighbors]
    local = labels[indices]
    counts = Counter(int(value) for value in local)
    agreement = max(counts.values()) / neighbors
    similarity = float(similarities[indices[0]])
    checks = {
        "similarity": similarity >= minimum_similarity,
        "local_label_agreement": agreement >= minimum_label_agreement,
        "classifier_confidence": classifier_confidence >= minimum_classifier_confidence,
        "committee_agreement": committee_agreement >= minimum_committee_agreement,
    }
    return {
        "abstain": not all(checks.values()),
        "checks": checks,
        "nearest_similarity": similarity,
        "local_label_agreement": agreement,
        "neighbor_indices": tuple(int(index) for index in indices),
    }


def cosine_support_quantile(training_embeddings: Any, quantile: float) -> float:
    """Fit a leave-self-out similarity threshold from training rows only."""

    matrix = np.asarray(training_embeddings, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2 or not 0.0 <= quantile <= 1.0:
        raise ValueError("support fitting needs a finite matrix and a valid quantile")
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / np.maximum(norms[:, None], 1e-15)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -math.inf)
    nearest = similarity.max(axis=1)
    return float(np.quantile(nearest, quantile, method="inverted_cdf"))


def correlation(predicted: Any, realized: Any) -> float:
    left = np.asarray(predicted, dtype=np.float64)
    right = np.asarray(realized, dtype=np.float64)
    if len(left) < 2 or np.std(left) <= 1e-15 or np.std(right) <= 1e-15:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else 0.0


def _standardize_fit(matrix: Any) -> Tuple[Any, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    return mean, np.where(scale > 1e-12, scale, 1.0)


def _weighted_ridge(
    matrix: Any, targets: Any, alpha: float, weights: Optional[Any] = None
) -> Mapping[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    response = np.asarray(targets, dtype=np.float64)
    mean, scale = _standardize_fit(values)
    standardized = (values - mean) / scale
    if response.ndim == 1:
        response = response[:, None]
    if weights is None:
        row_weights = np.ones(len(values), dtype=np.float64)
    else:
        row_weights = np.asarray(weights, dtype=np.float64)
    row_weights = row_weights / row_weights.mean()
    target_mean = np.average(response, axis=0, weights=row_weights)
    weighted_x = standardized * np.sqrt(row_weights)[:, None]
    weighted_y = (response - target_mean) * np.sqrt(row_weights)[:, None]
    system = weighted_x.T @ weighted_x + alpha * np.eye(weighted_x.shape[1])
    coefficients = np.linalg.solve(system, weighted_x.T @ weighted_y)
    return {
        "mean": mean,
        "scale": scale,
        "intercept": target_mean,
        "coefficients": coefficients,
        "alpha": float(alpha),
    }


def _ridge_predict(model: Mapping[str, Any], matrix: Any) -> Any:
    values = np.asarray(matrix, dtype=np.float64)
    result = (
        (values - model["mean"]) / model["scale"] @ model["coefficients"]
        + model["intercept"]
    )
    return result


def fit_event_utility_model(
    matrix: Any,
    gains: Any,
    *,
    classifier_l2: float,
    magnitude_ridge_l2: float,
) -> Mapping[str, Any]:
    """Fit fold-local three-way event and conditional magnitude heads."""

    features = np.asarray(matrix, dtype=np.float64)
    signed = np.asarray(gains, dtype=np.float64)
    labels = event_labels(np.zeros(len(signed)), signed)
    event_targets = np.eye(3, dtype=np.float64)[labels]
    event = _weighted_ridge(
        features,
        event_targets,
        classifier_l2,
        normalized_inverse_frequency_weights(labels),
    )
    magnitude = {}
    for name, label, values in (
        ("loss", 0, -signed),
        ("win", 2, signed),
    ):
        rows = labels == label
        magnitude[name] = (
            _weighted_ridge(features[rows], values[rows], magnitude_ridge_l2)
            if int(rows.sum()) >= 2
            else {"constant": float(values[rows].mean()) if rows.any() else 0.0}
        )
    return {
        "event": event,
        "magnitude": magnitude,
        "labels": labels,
        "classifier_l2": float(classifier_l2),
        "magnitude_ridge_l2": float(magnitude_ridge_l2),
        "calibration": {"method": "softmax", "temperature": 1.0},
    }


def predict_event_utility(model: Mapping[str, Any], matrix: Any) -> Mapping[str, Any]:
    """Return calibrated event probabilities and signed expected utility."""

    logits = _ridge_predict(model["event"], matrix)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    magnitudes = {}
    for name in ("loss", "win"):
        head = model["magnitude"][name]
        raw = (
            np.full(len(logits), head["constant"], dtype=np.float64)
            if "constant" in head
            else _ridge_predict(head, matrix).reshape(-1)
        )
        magnitudes[name] = np.maximum(raw, 0.0)
    utility = expected_signed_utility(
        probabilities, magnitudes["win"], magnitudes["loss"]
    )
    return {
        "probabilities": probabilities,
        "win_magnitude": magnitudes["win"],
        "loss_magnitude": magnitudes["loss"],
        "expected_utility": utility,
    }
