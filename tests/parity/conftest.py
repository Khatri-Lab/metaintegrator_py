"""Pytest fixtures for the R parity tests.

See ``parity_support.py`` for how golden sets are found and how to add more.
"""

from __future__ import annotations

import pytest

from metaintegrator.analysis import MetaAnalyzer

from .parity_support import ParityRun, discover_golden_dirs, run_python

_GOLDEN_DIRS = discover_golden_dirs()


@pytest.fixture(scope="session", params=_GOLDEN_DIRS, ids=[d.name for d in _GOLDEN_DIRS])
def parity_run(request) -> ParityRun:
    """Python results for one golden set (computed once per session)."""
    return run_python(request.param)


@pytest.fixture(scope="session")
def analyzer() -> MetaAnalyzer:
    """A single-core analyzer for filter tests."""
    return MetaAnalyzer(max_cores=1)
