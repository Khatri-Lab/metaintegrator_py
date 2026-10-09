"""Keep every place that states the package version in sync.

The version appears in four places: ``pyproject.toml``, the package's
``__version__``, ``CITATION.cff`` and the changelog. They drifted apart once
(0.1.0 in the package, 1.1.0 in the changelog, 1.0.0 in the citation).
This test stops that from happening again.

Versioning plan: every release before the first on-prem deployment is
0.x.x. The on-prem deployment is 1.0.0.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import metaintegrator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    if not path.is_file():
        pytest.skip(f"{relative_path} is not present in this checkout")
    return path.read_text(encoding="utf-8")


def _pyproject_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', _read("pyproject.toml"), re.MULTILINE)
    assert match, "pyproject.toml has no [project] version line"
    return match.group(1)


def test_dunder_version_matches_pyproject() -> None:
    assert metaintegrator.__version__ == _pyproject_version()


def test_citation_version_matches_pyproject() -> None:
    match = re.search(r"^version:\s*\"?([^\s\"]+)\"?\s*$", _read("CITATION.cff"), re.MULTILINE)
    assert match, "CITATION.cff has no version line"
    assert match.group(1) == _pyproject_version()


def test_changelog_has_an_entry_for_this_version() -> None:
    version = re.escape(_pyproject_version())
    assert re.search(rf"^## \[{version}\]", _read("docs/about/changelog.md"), re.MULTILINE)


def test_pre_deployment_versions_stay_below_one() -> None:
    """Releases before the on-prem deployment are 0.x.x.

    Change this test on purpose when cutting the 1.0.0 on-prem release.
    """
    major = int(_pyproject_version().split(".")[0])
    assert major == 0
