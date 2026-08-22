# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""The submitted container must execute the safe-margin router, not the
deliberately weak reference heuristic.

These checks run without Docker. They pin three separate things:

* ``container/entrypoint.py`` dispatches to ``baselines/safe_margin.py``;
* the public hash-regex artifact resolves from inside the runtime tree, so the
  evaluator's ``--input``/``--tier``/``--output`` invocation is enough;
* the ``container/Dockerfile`` ``COPY`` list, filtered through
  ``.dockerignore``, produces exactly the safe-margin runtime files and no
  development, public-data or build material.

The image-layout test stages a directory from that same derived file list and
runs the entry point out of it, so a missing runtime file fails here instead of
in the official evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
TOY_INPUT = ROOT / "data/toy/inputs.json"
ARTIFACT_PATH = ROOT / "baselines/hash-regex-public.v1.json"
TIERS = ("fast", "balanced", "premium")

#: Every path the submitted image is allowed to carry under ``/opt/router``.
EXPECTED_IMAGE_RUNTIME_FILES = (
    "baselines/hash-regex-public.v1.json",
    "baselines/hash_regex.py",
    "baselines/safe_margin.py",
    "entrypoint.py",
    "ossp_router/__init__.py",
    "ossp_router/cli.py",
    "ossp_router/heuristic.py",
    "ossp_router/image_evidence.py",
    "ossp_router/operator_helper.py",
    "ossp_router/orchestrator.py",
    "ossp_router/protocol.py",
    "ossp_router/resources/__init__.py",
    "ossp_router/resources/routing-policy.v1.json",
    "ossp_router/runtime.py",
    "ossp_router/scoring.py",
)

#: Repository material that must never reach the build context. Public inputs
#: and outcomes, the development tools and their tests, training sources,
#: unrelated baselines, build products, caches and credential-shaped files.
FORBIDDEN_CONTEXT_PATHS = (
    ".git/config",
    ".venv-data/bin/pip",
    "README.md",
    "baselines/README.md",
    "baselines/always_light.py",
    "baselines/hash-regex-public-dev-report.v1.json",
    "baselines/prompt_heuristic.py",
    "baselines/requirements-train.txt",
    "baselines/train_hash_regex.py",
    "build/mvp/report.json",
    "configs/routing-policy.v1.json",
    "container/.env",
    "container/README.md",
    "data/dev/outcomes.json",
    "data/materialized/dev/inputs.json",
    "data/public-data.v1.json",
    "data/toy/inputs.json",
    "data/train/outcomes.json",
    "docs/SCORING.md",
    "src/ossp_router/__pycache__/protocol.cpython-311.pyc",
    "src/ossp_router/public_runtime.py",
    "src/ossp_router/tiebreak_latency.py",
    "src/secret.key",
    "tests/test_safe_margin_router.py",
    "tools/run_mvp.py",
    "tools/stress_safe_margin.py",
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


entrypoint = _load_module("ossp_container_entrypoint", ROOT / "container/entrypoint.py")


def _segment_pattern(segment: str) -> str:
    """Translate one `.dockerignore` path segment into a regular expression."""

    parts = []
    for character in segment:
        if character == "*":
            parts.append("[^/]*")
        elif character == "?":
            parts.append("[^/]")
        else:
            parts.append(re.escape(character))
    return "".join(parts)


def _compile_dockerignore_pattern(pattern: str):
    """Compile one pattern, treating `**` as any number of path segments."""

    regex = ""
    for index, segment in enumerate(pattern.strip("/").split("/")):
        if segment == "**":
            regex += "(?:[^/]+/)*[^/]+" if index == 0 else "/(?:[^/]+/)*[^/]+"
            continue
        if index:
            regex += "/"
        regex += _segment_pattern(segment)
    return re.compile(regex + r"\Z")


def _dockerignore_rules():
    """Return (is_negation, compiled pattern) in file order."""

    rules = []
    for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        negated = stripped.startswith("!")
        body = stripped[1:] if negated else stripped
        rules.append((negated, _compile_dockerignore_pattern(body)))
    return tuple(rules)


def in_build_context(relative_path: str, rules) -> bool:
    """Apply Docker's last-match-wins `.dockerignore` semantics to one path."""

    included = True
    for negated, pattern in rules:
        if pattern.match(relative_path):
            included = negated
    return included


def _dockerfile_copy_instructions(text: str):
    """Return the (sources, destination) pairs of every `COPY` instruction."""

    joined = re.sub(r"\\\n\s*", " ", text)
    instructions = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        tokens = [
            token
            for token in stripped.split()[1:]
            if not token.startswith("--")
        ]
        instructions.append((tuple(tokens[:-1]), tokens[-1]))
    return tuple(instructions)


def _context_files_under(relative_directory: str, rules):
    """List the context-visible files below one repository directory."""

    base = ROOT / relative_directory
    found = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if in_build_context(relative, rules):
            found.append(relative)
    return found


def image_runtime_files(rules):
    """Derive the `/opt/router` file list from the Dockerfile and .dockerignore."""

    prefix = "/opt/router/"
    found = []
    for sources, destination in _dockerfile_copy_instructions(
        (ROOT / "container/Dockerfile").read_text(encoding="utf-8")
    ):
        assert destination.startswith(prefix) and destination.endswith("/")
        target = destination[len(prefix) :]
        for source in sources:
            if (ROOT / source).is_dir():
                # Docker copies a directory's *contents* into the destination.
                for relative in _context_files_under(source, rules):
                    found.append(target + relative[len(source) + 1 :])
                continue
            assert in_build_context(source, rules), source
            found.append(target + source.rsplit("/", 1)[-1])
    return tuple(sorted(found))


class DockerignoreMatcherTest(unittest.TestCase):
    """The matcher itself, so the context assertions below mean something."""

    def test_last_matching_rule_decides_inclusion(self) -> None:
        rules = tuple(
            (negated, _compile_dockerignore_pattern(body))
            for negated, body in (
                (False, "**"),
                (True, "src/"),
                (False, "src/**"),
                (True, "src/keep.py"),
            )
        )
        self.assertTrue(in_build_context("src/keep.py", rules))
        self.assertTrue(in_build_context("src", rules))
        self.assertFalse(in_build_context("src/drop.py", rules))
        self.assertFalse(in_build_context("src/nested/keep.py", rules))
        self.assertFalse(in_build_context("other.py", rules))

    def test_double_star_spans_any_number_of_segments(self) -> None:
        pattern = _compile_dockerignore_pattern("src/**")
        self.assertTrue(pattern.match("src/a"))
        self.assertTrue(pattern.match("src/a/b/c"))
        self.assertIsNone(pattern.match("src"))
        self.assertIsNone(pattern.match("other/a"))


class EntryPointDispatchTest(unittest.TestCase):
    """The official entry point must resolve to the safe-margin policy."""

    def test_entrypoint_no_longer_references_the_weak_reference_router(self) -> None:
        source = (ROOT / "container/entrypoint.py").read_text(encoding="utf-8")
        self.assertNotIn("ossp_router.heuristic", source)
        self.assertEqual("safe_margin", entrypoint.ROUTER_MODULE_NAME)

    def test_entrypoint_resolves_the_repository_safe_margin_module(self) -> None:
        self.assertEqual(ROOT / "baselines", entrypoint.resolve_router_directory())
        router_main = entrypoint.load_router_main()
        globals_ = router_main.__globals__
        self.assertEqual("safe-margin", globals_["STRATEGY_ID"])
        self.assertEqual(
            ROOT / "baselines/safe_margin.py",
            pathlib.Path(globals_["__file__"]).resolve(),
        )

    def test_missing_router_module_fails_with_the_documented_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            # Nested so that both candidate directories stay inside the
            # temporary tree and cannot match anything on this host.
            staged = pathlib.Path(temporary) / "runtime/entrypoint.py"
            staged.parent.mkdir()
            shutil.copyfile(ROOT / "container/entrypoint.py", staged)
            isolated = _load_module("ossp_entrypoint_isolated", staged)
            with self.assertRaises(FileNotFoundError):
                isolated.resolve_router_directory()
            self.assertEqual(
                2,
                isolated.main(["--input", "x", "--tier", "fast", "--output", "y"]),
            )


class BundledArtifactTest(unittest.TestCase):
    """The public artifact has to resolve without an evaluator argument."""

    def test_default_artifact_is_bundled_beside_the_router(self) -> None:
        safe_margin = sys.modules.get("safe_margin") or _load_module(
            "safe_margin", ROOT / "baselines/safe_margin.py"
        )
        self.assertEqual(ARTIFACT_PATH, safe_margin.DEFAULT_ARTIFACT_PATH)
        self.assertTrue(safe_margin.DEFAULT_ARTIFACT_PATH.is_file())

    def test_entrypoint_runs_with_no_artifact_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "submission.json"
            code = entrypoint.main(
                [
                    "--input",
                    str(TOY_INPUT),
                    "--tier",
                    "fast",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(0, code)
            self.assertEqual(0o644, stat.S_IMODE(output.stat().st_mode))
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual("fast", payload["tier"])
            self.assertEqual(
                len(json.loads(TOY_INPUT.read_text(encoding="utf-8"))["episodes"]),
                len(payload["decisions"]),
            )


class DockerBuildContextTest(unittest.TestCase):
    """The build context and the COPY list must match the safe-margin runtime."""

    @classmethod
    def setUpClass(cls):
        cls.rules = _dockerignore_rules()

    def test_dockerfile_copies_only_the_safe_margin_runtime_sources(self) -> None:
        self.assertEqual(
            (
                (("src", "container/entrypoint.py"), "/opt/router/"),
                (
                    (
                        "baselines/hash_regex.py",
                        "baselines/safe_margin.py",
                        "baselines/hash-regex-public.v1.json",
                    ),
                    "/opt/router/baselines/",
                ),
            ),
            _dockerfile_copy_instructions(
                (ROOT / "container/Dockerfile").read_text(encoding="utf-8")
            ),
        )

    def test_effective_context_holds_every_required_runtime_file(self) -> None:
        for relative in (
            "container/Dockerfile",
            "container/entrypoint.py",
            "baselines/hash-regex-public.v1.json",
            "baselines/hash_regex.py",
            "baselines/safe_margin.py",
            "src/ossp_router/heuristic.py",
            "src/ossp_router/protocol.py",
            "src/ossp_router/resources/routing-policy.v1.json",
        ):
            with self.subTest(path=relative):
                self.assertTrue(in_build_context(relative, self.rules))

    def test_effective_context_excludes_development_and_public_material(self) -> None:
        leaked = [
            relative
            for relative in FORBIDDEN_CONTEXT_PATHS
            if in_build_context(relative, self.rules)
        ]
        self.assertEqual([], leaked)

    def test_image_runtime_tree_is_exactly_the_expected_files(self) -> None:
        runtime = image_runtime_files(self.rules)
        self.assertEqual(EXPECTED_IMAGE_RUNTIME_FILES, runtime)
        # feature_budget.py stays whitelisted for container/measurement.Dockerfile
        # (the operator benchmark image) but must not enter the submitted image.
        self.assertNotIn("baselines/feature_budget.py", runtime)


class EmulatedOfficialInvocationTest(unittest.TestCase):
    """Stage the derived image tree and run the entry point out of it."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.runtime = pathlib.Path(cls.temporary.name) / "opt-router"
        for relative in image_runtime_files(_dockerignore_rules()):
            source = (
                ROOT / "container/entrypoint.py"
                if relative == "entrypoint.py"
                else ROOT / ("src/" + relative if relative.startswith("ossp_router/")
                             else relative)
            )
            target = cls.runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        cls.output_root = pathlib.Path(cls.temporary.name) / "out"
        cls.output_root.mkdir()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def _run(self, command, cwd):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(cwd)
        completed = subprocess.run(
            [sys.executable, *command],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(cwd),
            env=environment,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        return completed

    def test_image_layout_output_is_byte_identical_for_every_tier(self) -> None:
        for tier in TIERS:
            with self.subTest(tier=tier):
                emulated = self.output_root / f"entrypoint-{tier}.json"
                direct = self.output_root / f"direct-{tier}.json"
                self._run(
                    [
                        str(self.runtime / "entrypoint.py"),
                        "--input",
                        str(TOY_INPUT),
                        "--tier",
                        tier,
                        "--output",
                        str(emulated),
                    ],
                    cwd=self.runtime,
                )
                self._run(
                    [
                        str(ROOT / "baselines/safe_margin.py"),
                        "--input",
                        str(TOY_INPUT),
                        "--tier",
                        tier,
                        "--artifact",
                        str(ARTIFACT_PATH),
                        "--output",
                        str(direct),
                    ],
                    cwd=ROOT / "src",
                )
                self.assertEqual(direct.read_bytes(), emulated.read_bytes())
                self.assertEqual(
                    tier,
                    json.loads(emulated.read_text(encoding="utf-8"))["tier"],
                )

    def test_repository_layout_entrypoint_matches_the_image_layout(self) -> None:
        image_output = self.output_root / "layout-image.json"
        repository_output = self.output_root / "layout-repository.json"
        self._run(
            [
                str(self.runtime / "entrypoint.py"),
                "--input",
                str(TOY_INPUT),
                "--tier",
                "premium",
                "--output",
                str(image_output),
            ],
            cwd=self.runtime,
        )
        self._run(
            [
                str(ROOT / "container/entrypoint.py"),
                "--input",
                str(TOY_INPUT),
                "--tier",
                "premium",
                "--output",
                str(repository_output),
            ],
            cwd=ROOT / "src",
        )
        self.assertEqual(
            image_output.read_bytes(), repository_output.read_bytes()
        )


if __name__ == "__main__":
    unittest.main()
