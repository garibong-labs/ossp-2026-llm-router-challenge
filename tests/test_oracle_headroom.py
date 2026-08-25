# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Focused checks for the development-only oracle headroom analysis."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "tools"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


oracle_headroom = _load_module(
    "oracle_headroom", ROOT / "tools/oracle_headroom.py"
)


class _FakeData:
    """Minimal SplitData stand-in with a realized cost/score matrix."""

    def __init__(self, rows):
        model_ids = ("ax31-light", "ax31", "axk1-think")
        self.costs = tuple(
            {model_ids[j]: row[0][j] for j in range(3)} for row in rows
        )
        self.scores = tuple(
            {model_ids[j]: row[1][j] for j in range(3)} for row in rows
        )
        self.num_episodes = len(rows)


class OracleHullTest(unittest.TestCase):
    def test_a_dominated_middle_model_is_skipped_on_the_hull(self):
        # ax31 costs more than think but scores less: the hull must jump
        # straight from light to think.
        data = _FakeData([((1.0, 5.0, 4.0), (0.0, 0.2, 1.0))])
        base, steps = oracle_headroom.oracle_base_and_steps(data)
        self.assertEqual([0], list(base))
        self.assertEqual(1, len(steps))
        self.assertEqual(2, steps[0].model_index)

    def test_a_cheaper_better_model_becomes_the_free_base(self):
        data = _FakeData([((1.0, 0.9, 30.0), (0.0, 1.0, 1.0))])
        base, steps = oracle_headroom.oracle_base_and_steps(data)
        self.assertEqual([1], list(base))
        self.assertEqual([], list(steps))

    def test_steps_are_ordered_by_decreasing_efficiency(self):
        data = _FakeData(
            [
                ((1.0, 2.0, 3.0), (0.0, 0.5, 0.6)),
                ((1.0, 3.0, 9.0), (0.0, 0.1, 0.2)),
            ]
        )
        _base, steps = oracle_headroom.oracle_base_and_steps(data)
        efficiencies = [step.efficiency for step in steps]
        self.assertEqual(sorted(efficiencies, reverse=True), efficiencies)

    def test_greedy_selection_respects_the_budget(self):
        data = _FakeData(
            [
                ((1.0, 2.0, 10.0), (0.0, 1.0, 1.0)),
                ((1.0, 2.0, 10.0), (0.0, 1.0, 1.0)),
            ]
        )
        base, steps = oracle_headroom.oracle_base_and_steps(data)
        # Budget 1.75x light total: the first upgrade (+1.0) fits, the second
        # does not, leaving a strictly positive fractional LP remainder.
        selection, bonus = oracle_headroom.greedy_selection(
            data, base, steps, 1.75
        )
        self.assertEqual(1, sum(selection))
        self.assertGreater(bonus, 0.0)

    def test_weighted_score_uses_the_published_tier_weights(self):
        value = oracle_headroom.weighted_score(
            {"fast": 1.0, "balanced": 0.0, "premium": 0.0}
        )
        self.assertAlmostEqual(0.4, value)
        value = oracle_headroom.weighted_score(
            {"fast": 0.0, "balanced": 1.0, "premium": 1.0}
        )
        self.assertAlmostEqual(0.6, value)

    def test_stop_gate_target_is_frozen(self):
        self.assertEqual(0.690000, oracle_headroom.STOP_GATE_TARGET)


class DevelopmentIsolationTest(unittest.TestCase):
    """Oracle analysis must be unreachable from the submitted runtime."""

    def test_runtime_and_baselines_never_reference_the_oracle_tools(self):
        runtime_paths = [
            *sorted((ROOT / "src/ossp_router").glob("*.py")),
            *sorted((ROOT / "baselines").glob("*.py")),
            ROOT / "container/entrypoint.py",
        ]
        for path in runtime_paths:
            if path.name == "train_risk_calibrated.py":
                # Training tool is development-only and never shipped.
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn(
                "oracle_headroom",
                text,
                f"{path.name} must not reference the oracle tool",
            )

    def test_oracle_tool_is_not_part_of_the_container_image(self):
        dockerfile = (ROOT / "container/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("tools/", dockerfile)
        self.assertNotIn("oracle", dockerfile)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
