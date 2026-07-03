# ── Path bootstrap ────────────────────────────────────────────────────
# Ensure that ``src/`` is on sys.path so that cross-package bare absolute
# imports (e.g. ``from preprocessing.xxx import …``) resolve correctly
# regardless of how the code is invoked:
#   • python -m src            (CWD on path, but NOT src/)
#   • pytest                   (conftest.py also adds src/)
#   • Jupyter notebooks         (import src → this file runs)
#   • PYTHONPATH=src/ python … (already on path, no-op here)
import sys as _sys
from pathlib import Path as _Path

_src_dir = str(_Path(__file__).resolve().parent)
if _src_dir not in _sys.path:
    _sys.path.insert(0, _src_dir)

del _sys, _Path, _src_dir
