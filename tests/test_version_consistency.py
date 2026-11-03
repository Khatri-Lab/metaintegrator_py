"""Keep every place that states the package version in sync.

The version appears in four places: ``pyproject.toml``, the package's
``__version__``, ``CITATION.cff`` and the changelog. They drifted apart once
(0.1.0 in the package, 1.1.0 in the changelog, 1.0.0 in the citation).
This test stops that from happening again.

Versioning plan: every release before the first on-prem deployment was
0.x.x. The on-prem deployment, on 2026-11-04, is 1.0.0. Later releases
follow Semantic Versioning from there.
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


def test_citation_date_matches_changelog_date() -> None:
    """``date-released`` is the date in this version's changelog heading."""
    version = re.escape(_pyproject_version())
    heading = re.search(
        rf"^## \[{version}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        _read("docs/about/changelog.md"),
        re.MULTILINE,
    )
    assert heading, "the changelog heading for this version has no release date"
    released = re.search(
        r"^date-released:\s*\"?([\d-]+)\"?\s*$", _read("CITATION.cff"), re.MULTILINE
    )
    assert released, "CITATION.cff has no date-released line"
    assert released.group(1) == heading.group(1)


def test_major_version_is_one_from_the_on_prem_release() -> None:
    """The on-prem release made the major version 1.

    Releases before it were 0.x.x. Major version 1 is now the expected one.
    A move to 2.0.0 means a breaking change. Change this test on purpose
    when that happens.
    """
    major = int(_pyproject_version().split(".")[0])
    assert major == 1


def test_version_has_three_numeric_parts() -> None:
    """``MAJOR.MINOR.PATCH``, as in the changelog headings and the tags."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", _pyproject_version())
