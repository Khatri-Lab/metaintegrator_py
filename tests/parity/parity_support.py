"""Support code for the R parity tests: golden discovery and cached runs.

The parity tests replay R MetaIntegrator inputs through metaintegrator_py.
They then compare the Python results with R's own, frozen outputs.

Golden sets
-----------
A golden set is one R ``metaObject`` exported to CSV with
``src/utils/robust_rdata_converter.R``. See ``src/utils/r_golden.py``.

- The ``tinyMetaObject`` that ships with R MetaIntegrator is always used.
  It lives in ``data/reference/metaintegrator/input/tinyMetaObject``.
- More golden sets can be added without code changes. Point the
  ``MIPY_PARITY_GOLDEN_DIRS`` environment variable at one or more golden
  directories, separated by ``os.pathsep`` (``:`` on macOS and Linux).

Each golden set is analyzed once per test session and shared by all tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from metaintegrator.analysis import MetaAnalyzer
from utils.r_golden import RGoldenMetaObject, load_r_golden

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Always-available golden set: R MetaIntegrator's built-in tinyMetaObject.
BUILTIN_GOLDEN_DIRS: List[Path] = [
    PROJECT_ROOT / "data" / "reference" / "metaintegrator" / "input" / "tinyMetaObject",
]

GOLDEN_DIRS_ENV = "MIPY_PARITY_GOLDEN_DIRS"

# Tolerances for numeric parity. Python and R agree to about 1e-12 today,
# so these leave headroom for platform-level floating-point noise while
# still catching any real algorithm change.
RTOL = 1e-8
ATOL = 1e-10


def discover_golden_dirs() -> List[Path]:
    """Return the built-in golden directories plus any listed in the env var.

    A directory listed in ``MIPY_PARITY_GOLDEN_DIRS`` that does not exist is
    an error, not a silent skip. A typo would otherwise turn off the tests
    without anyone noticing.

    Returns:
        Golden directories without duplicates, built-ins first.

    Raises:
        FileNotFoundError: If a directory listed in the env var is missing.
    """
    extra = [
        Path(p).expanduser()
        for p in os.environ.get(GOLDEN_DIRS_ENV, "").split(os.pathsep)
        if p.strip()
    ]
    missing = [str(d) for d in extra if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"{GOLDEN_DIRS_ENV} lists missing directories: {missing}")

    unique: List[Path] = []
    for d in BUILTIN_GOLDEN_DIRS + extra:
        if d.is_dir() and d.resolve() not in {u.resolve() for u in unique}:
            unique.append(d)
    return unique


@dataclass
class ParityRun:
    """One golden set together with the Python results computed from it.

    Attributes:
        golden: The R golden object.
        meta_object: The Python ``meta_object`` after ``run_meta_analysis()``.
        name_map: R dataset name to the (possibly converted) Python name.
    """

    golden: RGoldenMetaObject
    meta_object: Dict
    name_map: Dict[str, str]

    @property
    def meta_analysis(self) -> Dict:
        """Python's ``metaAnalysis`` result."""
        return self.meta_object["metaAnalysis"]

    def python_loo_pooled(self, r_dataset_name: str):
        """Python leave-one-out ``pooledResults`` with *r_dataset_name* removed."""
        loo = self.meta_object.get("leaveOneOutAnalysis") or {}
        result = loo.get(f"removed_{self.name_map[r_dataset_name]}")
        return None if result is None else result["pooledResults"]


_RUN_CACHE: Dict[Path, ParityRun] = {}


def run_python(golden_dir: Path) -> ParityRun:
    """Load a golden set and run the Python meta-analysis on its inputs."""
    if golden_dir not in _RUN_CACHE:
        golden = load_r_golden(golden_dir)
        meta_object = golden.to_meta_object()
        # One core keeps the run deterministic and avoids process pools in CI.
        analyzer = MetaAnalyzer(max_cores=1)
        meta_object = analyzer.run_meta_analysis(
            meta_object, run_leave_one_out=bool(golden.leave_one_out)
        )
        # run_meta_analysis() may rename datasets (for example dots to
        # underscores) but keeps their order.
        name_map = dict(zip(golden.dataset_names, meta_object["originalData"].keys()))
        _RUN_CACHE[golden_dir] = ParityRun(golden, meta_object, name_map)
    return _RUN_CACHE[golden_dir]
