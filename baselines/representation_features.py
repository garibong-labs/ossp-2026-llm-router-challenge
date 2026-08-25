# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Bounded, prompt-only representations for the family-disjoint audit.

This module deliberately uses only the Python standard library and fields that
exist in the runtime ``Episode`` protocol.  It does not inspect episode IDs,
split names, source labels, positions, outcomes, or any external resource.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Sequence, Tuple

from ossp_router.heuristic import episode_text
from ossp_router.protocol import Episode, Message

import hash_regex


SEMANTIC_HASH_BINS = 256
MAX_FIELD_CHARACTERS = 32_768
_WORD = re.compile(r"[A-Za-z]+|[\uac00-\ud7a3]+|\d+", re.UNICODE)
_SENTENCE = re.compile(r"[.!?\u3002\uff01\uff1f]")
_OPTION = re.compile(r"(?:^|\n)\s*(?:[A-Ha-h]|\d{1,2})[.)]\s+", re.MULTILINE)
_QUESTION_BOUNDARY = re.compile(
    r"(?:^|\n)\s*(?:question|q|problem|query|instruction|\ubb38\uc81c|\uc9c8\ubb38|\uc694\uccad)\s*[:：]",
    re.IGNORECASE,
)


EXPANDED_STRUCTURAL_FEATURE_NAMES = (
    "log_character_count",
    "log_word_count",
    "log_sentence_count",
    "log_field_count",
    "log_system_field_count",
    "log_user_field_count",
    "log_assistant_field_count",
    "log_other_field_count",
    "system_character_fraction",
    "user_character_fraction",
    "assistant_character_fraction",
    "longest_field_fraction",
    "first_field_fraction",
    "last_field_fraction",
    "log_newline_count",
    "log_paragraph_count",
    "log_question_mark_count",
    "log_option_count",
    "log_code_fence_count",
    "log_indented_line_count",
    "log_equation_line_count",
    "numeric_fraction",
    "uppercase_fraction",
    "hangul_fraction",
    "punctuation_fraction",
    "unique_word_fraction",
    "mean_word_length",
    "context_fraction",
    "log_context_characters",
    "log_question_characters",
    "has_context_question_boundary",
    "has_explicit_system_field",
    "has_role_transition",
    "log_parenthesis_count",
    "log_bracket_count",
    "log_quote_count",
)


def _fields(episode: Episode) -> Tuple[Tuple[str, str], ...]:
    if episode.prompt is not None:
        return (("prompt", episode.prompt),)
    assert episode.messages is not None
    return tuple((message.role.casefold(), message.content) for message in episode.messages)


def _bounded_episode(episode: Episode) -> Episode:
    """Return the prompt/message view used by bounded candidate features."""

    if episode.prompt is not None:
        return Episode(
            episode.episode_id,
            prompt=episode.prompt[:MAX_FIELD_CHARACTERS],
        )
    assert episode.messages is not None
    return Episode(
        episode.episode_id,
        messages=tuple(
            Message(message.role, message.content[:MAX_FIELD_CHARACTERS])
            for message in episode.messages
        ),
    )


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / max(1.0, denominator)


def expanded_structural_vector(episode: Episode) -> Tuple[float, ...]:
    """Return field/role-aware shape and derivable context/question features."""

    episode = _bounded_episode(episode)
    fields = _fields(episode)
    text = episode_text(episode)
    characters = len(text)
    words = _WORD.findall(text)
    nonspace = sum(not char.isspace() for char in text)
    role_counts = {
        "system": sum(role == "system" for role, _ in fields),
        "user": sum(role in ("user", "prompt") for role, _ in fields),
        "assistant": sum(role == "assistant" for role, _ in fields),
    }
    role_characters = {
        role: sum(len(content) for field_role, content in fields if field_role == role)
        for role in ("system", "assistant")
    }
    role_characters["user"] = sum(
        len(content) for role, content in fields if role in ("user", "prompt")
    )
    boundary = None
    for match in _QUESTION_BOUNDARY.finditer(text):
        boundary = match.start()
    if boundary is None:
        context_characters = 0
        question_characters = characters
    else:
        context_characters = boundary
        question_characters = characters - boundary
    field_lengths = [len(content) for _role, content in fields]
    punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
    uppercase = sum(char.isupper() for char in text)
    hangul = sum("\uac00" <= char <= "\ud7a3" for char in text)
    numeric = sum(char.isdecimal() for char in text)
    unique_words = len({word.casefold() for word in words})
    equation_lines = sum(
        bool(re.search(r"[=+*/^]|\\(?:frac|sum|int|sqrt)\b", line))
        for line in text.splitlines()
    )
    transitions = sum(
        left[0] != right[0] for left, right in zip(fields, fields[1:])
    )
    return (
        math.log1p(characters),
        math.log1p(len(words)),
        math.log1p(max(1, len(_SENTENCE.findall(text)))),
        math.log1p(len(fields)),
        math.log1p(role_counts["system"]),
        math.log1p(role_counts["user"]),
        math.log1p(role_counts["assistant"]),
        math.log1p(len(fields) - sum(role_counts.values())),
        _ratio(role_characters["system"], characters),
        _ratio(role_characters["user"], characters),
        _ratio(role_characters["assistant"], characters),
        _ratio(max(field_lengths, default=0), characters),
        _ratio(field_lengths[0] if field_lengths else 0, characters),
        _ratio(field_lengths[-1] if field_lengths else 0, characters),
        math.log1p(text.count("\n")),
        math.log1p(len(re.findall(r"\n\s*\n", text))),
        math.log1p(text.count("?") + text.count("\uff1f")),
        math.log1p(len(_OPTION.findall(text))),
        math.log1p(text.count("```")),
        math.log1p(sum(line.startswith(("    ", "\t")) for line in text.splitlines())),
        math.log1p(equation_lines),
        _ratio(numeric, nonspace),
        _ratio(uppercase, nonspace),
        _ratio(hangul, nonspace),
        _ratio(punctuation, nonspace),
        _ratio(unique_words, len(words)),
        _ratio(sum(len(word) for word in words), len(words)),
        _ratio(context_characters, characters),
        math.log1p(context_characters),
        math.log1p(question_characters),
        float(boundary is not None),
        float(role_counts["system"] > 0),
        float(transitions > 0),
        math.log1p(text.count("(") + text.count(")")),
        math.log1p(text.count("[") + text.count("]")),
        math.log1p(text.count('"') + text.count("'")),
    )


def _normalized_words(text: str) -> Tuple[str, ...]:
    result = []
    for value in _WORD.findall(text[:MAX_FIELD_CHARACTERS]):
        token = value.casefold()
        result.append("<number>" if token.isdecimal() else token)
    return tuple(result)


def _semantic_items(episode: Episode) -> Iterable[str]:
    """Yield bounded field-aware word and character n-grams."""

    for role, raw in _fields(episode):
        role = role if role in ("system", "user", "assistant", "tool") else "prompt"
        text = raw[:MAX_FIELD_CHARACTERS]
        words = _normalized_words(text)
        for word in words:
            yield f"{role}:w1:{word}"
        for left, right in zip(words, words[1:]):
            yield f"{role}:w2:{left}\x1f{right}"
        compact = " ".join(text.casefold().split())
        for width in (3, 4, 5):
            for index in range(max(0, len(compact) - width + 1)):
                yield f"{role}:c{width}:{compact[index:index + width]}"


def semantic_proxy_vector(
    episode: Episode, hash_bins: int = SEMANTIC_HASH_BINS
) -> Tuple[float, ...]:
    """Return signed field-aware word/character n-gram hash bins."""

    if hash_bins < 16 or hash_bins > 16_384 or hash_bins & (hash_bins - 1):
        raise ValueError("semantic hash bins must be a power of two in [16, 16384]")
    bins = [0.0] * hash_bins
    for value in _semantic_items(episode):
        digest = hash_regex._stable_hash(value)
        bins[digest & (hash_bins - 1)] += -1.0 if digest & (1 << 63) else 1.0
    norm = math.sqrt(sum(value * value for value in bins))
    if norm:
        bins = [value / norm for value in bins]
    return tuple(bins)


def representation_vector(
    episode: Episode, name: str, *, semantic_bins: int = SEMANTIC_HASH_BINS
) -> Tuple[float, ...]:
    """Build one predeclared audit representation by name."""

    if name == "A-current-dense-wordhash":
        return hash_regex.raw_feature_vector(episode, hash_regex.DEFAULT_HASH_BINS)
    structural = expanded_structural_vector(episode)
    if name == "B-expanded-structural":
        return structural
    semantic = semantic_proxy_vector(episode, semantic_bins)
    if name == "C-semantic-proxy":
        bounded = _bounded_episode(episode)
        dense = hash_regex.raw_feature_vector(bounded, 256)[
            : len(hash_regex.DENSE_FEATURE_NAMES)
        ]
        return dense + semantic
    if name == "D-structural-semantic":
        return structural + semantic
    raise ValueError(f"unknown representation: {name}")


REPRESENTATIONS = (
    "A-current-dense-wordhash",
    "B-expanded-structural",
    "C-semantic-proxy",
    "D-structural-semantic",
)


def field_character_bound(name: str) -> Optional[int]:
    """Return the enforced per-field bound, or ``None`` for exact reference A."""

    if name not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {name}")
    return None if name == REPRESENTATIONS[0] else MAX_FIELD_CHARACTERS
