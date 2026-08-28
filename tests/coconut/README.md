# COCONUT Test Suite — Running Instructions

Tests for `src/coconut/coconut.py` (the COCONUT batch-normalization algorithm)
and `src/coconut/coconut_runner.py` (its CSV-loading/CLI wrapper).

> This directory is part of the metaintegrator_py package and shares its root
> `pyproject.toml` config, `requirements-dev.txt`, and test-running conventions.

## Test files

| File | What it covers |
|---|---|
| `test_coconut.py` | `coconut()` and its internals — data fixtures, ComBat helpers, iterative solver, non-parametric prior, integration tests, a couple of debug/perf tests. |
| `test_coconut_runner.py` | `coconut_runner.py` — CSV discovery/loading, dataset name extraction, output writing, the CLI entry point. |
| `test_runner.py` | **Not a pytest test module** despite the `test_` prefix — a standalone CLI script (`python test_runner.py [all\|unit\|integration\|fast\|performance\|coverage]`) that wraps `pytest` invocations with preset flag combinations. Pytest will import it during collection (harmless — it defines no `test_*`/`Test*` items, so nothing runs from it) but you run it directly with `python`, not `pytest`. |
| `__init__.py` | Empty; makes this a package so pytest's default import mode doesn't collide it with same-named test modules elsewhere in the repo. |

There is no `pytest.ini` here — pytest configuration for the whole project
(including this directory) lives in the repo root's `pyproject.toml`
(`[tool.pytest.ini_options]`).

## Setup

Install the project's dev/test dependencies from the **repo root** (there is
no separate dependency file for this directory):

```bash
cd metaintegrator_py
pip install -r requirements-dev.txt
pip install -e ".[test]"
```

The root `pyproject.toml` sets `pythonpath = ["src"]` for pytest, so
`coconut`, `metaintegrator`, etc. are importable during test runs without
any extra setup. `test_coconut.py` and `test_coconut_runner.py` also each
insert `../../src` onto `sys.path` manually at import time — redundant
under pytest given the config above, but it's what makes them importable if
you ever run one of them directly as a plain script instead.

## Running the tests

Run from the **repo root** (recommended — matches `pyproject.toml`'s
`testpaths` and coverage `source` config) or from this directory; both work:

```bash
# From the repo root
pytest tests/coconut/ -v

# From tests/coconut/
pytest test_coconut.py test_coconut_runner.py -v
```

Specific tests, using the class/method names that actually exist today:

```bash
pytest tests/coconut/test_coconut.py::test_debug_coconut_steps -v -s
pytest tests/coconut/test_coconut.py::TestCoconutMainFunction::test_data_structure_validation -v -s
pytest tests/coconut/test_coconut.py::TestCoconutMainFunction::test_simple_coconut_functionality -v -s
pytest tests/coconut/test_coconut.py::TestIntegrationTests -v
pytest tests/coconut/test_coconut_runner.py::TestIntegration -v
```

Note `test_debug_coconut_steps` is a module-level function, not a method of
`TestCoconutMainFunction` — don't qualify it under that class or pytest
won't find it.

With coverage:

```bash
pytest tests/coconut/ --cov=coconut --cov-report=html
```

### Markers

The root `pyproject.toml` registers seven markers project-wide: `slow`,
`integration`, `unit`, `parametric`, `nonparametric`, `performance`, and
`regression`. Of those, **only `slow` is actually applied** anywhere in this
directory today (one test in `test_coconut.py`, `TestPerformanceAndStress`).
`integration`/`unit` etc. aren't decorated onto any test here — the
`TestIntegrationTests` / `TestIntegration` classes above are just class
*names*, not `@pytest.mark.integration`. So:

```bash
pytest tests/coconut/ -m "not slow"      # skips the perf/stress test — does something
pytest tests/coconut/ -m "not integration"  # currently a no-op here — nothing's marked
```

is accurate; a command like `-m "not integration and not slow"` only
actually filters on the `slow` half of that expression today, since nothing
here is marked `integration`.

## Debugging tips

- Run the debug test first (`test_debug_coconut_steps`, `-v -s` to see its
  print output) — it walks through COCONUT's steps manually and is usually
  the fastest way to localize a data-shape/indexing problem.
- If you see `ImportWarning: Could not import COCONUT functions...` followed
  by `NotImplementedError: ... not imported. Please fix imports.`, that's
  `test_coconut.py`'s own fallback path (see its `try`/`except ImportError`
  block) — it means the `coconut` package genuinely wasn't importable
  (usually a missing `pip install -e .` from the repo root, or running from
  an environment that doesn't have `src/` reachable), not a bug in the test
  itself.
- Large-dataset tests are marked `slow`; memory tests need `psutil`
  (already in `requirements-dev.txt`); coverage needs `pytest-cov` (same).

## Running in Docker

If you're using the metaintegrator_py dev container set up in
`metaintegrator-web/docs/docker_mac_host_dev_setup.md`, these tests run the
same way inside it:

```bash
make mipy-shell   # from the metaintegrator-web repo root
pytest tests/coconut/ -v
```
