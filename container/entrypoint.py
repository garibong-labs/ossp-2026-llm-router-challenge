# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-FileCopyrightText: Copyright 2026 OSSP 2026 LLM Router Challenge participant
# SPDX-License-Identifier: Apache-2.0

"""Container entry point for one-tier prompt routing.

The submitted image runs the safe-margin router in ``baselines/safe_margin.py``.
That module is the single authoritative policy implementation: the development
runner (``tools/run_mvp.py``), the composition stress tool
(``tools/stress_safe_margin.py``) and this entry point all execute the same
code, so the container cannot drift from the measured public Train/Dev
decisions.

Only the module lookup lives here. ``safe_margin.main`` keeps the documented
``router-run`` interface (``--input``, ``--tier``, ``--output`` and an optional
``--policy``) and defaults ``--artifact`` to the public hash-regex artifact
bundled beside it, so the evaluator never passes an extra argument.

Two directory layouts have to resolve:

``<runtime>/baselines/safe_margin.py``
    the submitted image, where ``container/Dockerfile`` copies ``src`` and this
    file into ``/opt/router/`` and the router files into
    ``/opt/router/baselines/``;
``<repository>/baselines/safe_margin.py``
    a plain checkout, so the official entry point can be exercised without
    Docker.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence


_ENTRYPOINT_DIRECTORY = Path(__file__).resolve().parent

#: Checked in order; the first directory holding the router module wins.
ROUTER_DIRECTORY_CANDIDATES = (
    _ENTRYPOINT_DIRECTORY / "baselines",
    _ENTRYPOINT_DIRECTORY.parent / "baselines",
)
ROUTER_MODULE_NAME = "safe_margin"


def resolve_router_directory() -> Path:
    """Return the bundled directory that holds the safe-margin router."""

    for candidate in ROUTER_DIRECTORY_CANDIDATES:
        if (candidate / f"{ROUTER_MODULE_NAME}.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"{ROUTER_MODULE_NAME} 라우터 모듈을 찾을 수 없습니다: "
        + ", ".join(str(candidate) for candidate in ROUTER_DIRECTORY_CANDIDATES)
    )


def load_router_main():
    """Import the authoritative ``main`` instead of restating the policy."""

    directory = str(resolve_router_directory())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    from safe_margin import main

    return main


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        router_main = load_router_main()
    except (ImportError, OSError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    return router_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
