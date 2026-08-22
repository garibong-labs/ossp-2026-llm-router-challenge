# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Checks for the development-only one-command MVP runner."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import unittest

from ossp_router.protocol import MODEL_IDS, TIERS


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOY_INPUT = ROOT / "data/toy/inputs.json"
TOY_OUTCOMES = ROOT / "data/toy/outcomes.json"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_mvp = _load_module("run_mvp", ROOT / "tools/run_mvp.py")


def _run(target, extra=()):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = run_mvp.main(
            [
                "--input",
                str(TOY_INPUT),
                "--outcomes",
                str(TOY_OUTCOMES),
                "--output-dir",
                str(target),
                *extra,
            ]
        )
    return code, buffer.getvalue()


class RunMvpTest(unittest.TestCase):
    def test_runner_writes_three_tier_files_and_a_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "mvp"
            code, _output = _run(target)
            self.assertEqual(0, code)
            for tier in TIERS:
                with self.subTest(tier=tier):
                    path = target / f"{tier}.json"
                    self.assertTrue(path.is_file())
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(tier, payload["tier"])
                    self.assertEqual(
                        sorted(
                            item["episode_id"] for item in payload["decisions"]
                        ),
                        sorted(
                            {item["episode_id"] for item in payload["decisions"]}
                        ),
                    )
            self.assertTrue((target / "report.json").is_file())

    def test_report_records_metrics_plans_and_baseline_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "mvp"
            self.assertEqual(0, _run(target)[0])
            report = json.loads(
                (target / "report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                "safe-margin-mvp-development-report", report["report_type"]
            )
            self.assertEqual("safe-margin", report["strategy"])
            self.assertIn("final_score", report["score"])
            for tier in TIERS:
                with self.subTest(tier=tier):
                    scored = report["score"]["tiers"][tier]
                    self.assertIn("budget_ratio", scored)
                    self.assertIn("quality_score", scored)
                    self.assertTrue(scored["budget_passed"])
                    plan = report["tier_plans"][tier]
                    self.assertEqual(
                        set(MODEL_IDS), set(plan["model_counts"])
                    )
                    self.assertLessEqual(
                        plan["predicted_budget_ratio"],
                        plan["target_ratio"] + 1e-9,
                    )
                    self.assertTrue(plan["stages"])
            for baseline in (
                "all-light",
                "prompt-heuristic",
                "feature-budget",
                "hash-regex",
            ):
                self.assertIn(baseline, report["public_dev_baselines"])

    def test_printed_summary_compares_against_the_published_baselines(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "mvp"
            code, output = _run(target)
            self.assertEqual(0, code)
            for needle in (
                "all-light",
                "hash-regex",
                "safe-margin (MVP)",
                "weighted final score",
                "선택 분포",
            ):
                with self.subTest(needle=needle):
                    self.assertIn(needle, output)

    def test_report_never_stores_a_local_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "mvp"
            self.assertEqual(0, _run(target)[0])
            text = (target / "report.json").read_text(encoding="utf-8")
            self.assertNotIn(temporary, text)
            self.assertNotIn(str(ROOT), text)
            report = json.loads(text)
            self.assertEqual("data/toy/inputs.json", report["input_path"])
            self.assertEqual(
                "baselines/hash-regex-public.v1.json", report["artifact_path"]
            )

    def test_runner_output_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = pathlib.Path(temporary) / "first"
            second = pathlib.Path(temporary) / "second"
            self.assertEqual(0, _run(first)[0])
            self.assertEqual(0, _run(second)[0])
            for tier in TIERS:
                with self.subTest(tier=tier):
                    self.assertEqual(
                        (first / f"{tier}.json").read_bytes(),
                        (second / f"{tier}.json").read_bytes(),
                    )
            self.assertEqual(
                (first / "report.json").read_bytes(),
                (second / "report.json").read_bytes(),
            )

    def test_missing_public_input_is_reported_without_masking(self):
        with tempfile.TemporaryDirectory() as temporary:
            absent = pathlib.Path(temporary) / "absent.json"
            with self.assertRaises(run_mvp.MvpRunError) as caught:
                run_mvp.discover_split_paths("dev", absent, TOY_OUTCOMES)
            self.assertIn("materialize_public_data.py", str(caught.exception))

    def test_main_returns_two_for_an_unusable_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = pathlib.Path(temporary) / "mvp"
            buffer = io.StringIO()
            with contextlib.redirect_stderr(buffer):
                code = run_mvp.main(
                    [
                        "--input",
                        str(pathlib.Path(temporary) / "absent.json"),
                        "--outcomes",
                        str(TOY_OUTCOMES),
                        "--output-dir",
                        str(target),
                    ]
                )
            self.assertEqual(2, code)
            self.assertFalse((target / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
