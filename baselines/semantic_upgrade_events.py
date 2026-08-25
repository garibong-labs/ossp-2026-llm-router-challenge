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
from collections import Counter
from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from ossp_router.protocol import Episode


MAX_FIELD_CHARACTERS = 32_768
EVENTS = ("loss", "tie", "win")
STEPS = ("ax31-light->ax31", "ax31->axk1-think")


def bounded_role_content(episode: Episode) -> Tuple[Tuple[str, str], ...]:
    """Return only bounded runtime-authorized role/content fields."""

    if episode.prompt is not None:
        return (("user", episode.prompt[:MAX_FIELD_CHARACTERS]),)
    assert episode.messages is not None
    return tuple(
        (message.role, message.content[:MAX_FIELD_CHARACTERS])
        for message in episode.messages
    )


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
