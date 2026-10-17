# /// script
# requires-python = ">=3.12"
# dependencies = ["hatchling==1.32.4"]
# ///
"""Build the Python-only universal wheel into a directory and print its path.

Run as ``uv run scripts/build_pure_wheel.py OUTPUT_DIRECTORY``. The project
builds with maturin; hatchling, called directly, reads the same ``[project]``
table and ``[tool.hatch.build.targets.wheel]`` and ignores ``[build-system]``.
Hatchling is pinned to the version in ``uv.lock``, so the release builds with
the one the tests use; ``pure_wheel_test`` checks the two agree.
"""

from __future__ import annotations

from pathlib import Path

import os
import sys

from hatchling.build import build_wheel


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Build the wheel from the repository root."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: build_pure_wheel.py OUTPUT_DIRECTORY")
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Hatchling reads the project from the working directory.
    os.chdir(ROOT)
    print(output / build_wheel(str(output)))


if __name__ == "__main__":
    main()
