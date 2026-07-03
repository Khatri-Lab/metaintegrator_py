# conftest.py  (project root)
# ──────────────────────────────────────────────────────────────────────
# Ensure ``src/`` is on sys.path so that bare absolute cross-package
# imports (e.g. ``from preprocessing.xxx``) resolve when running pytest
# from the project root, even without setting PYTHONPATH.
#
# This mirrors the bootstrap in ``src/__init__.py`` and guarantees the
# same import semantics in CLI, pytest, and notebook contexts.
# ──────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
