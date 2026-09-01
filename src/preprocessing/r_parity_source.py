"""
Backward-compatibility alias: use :mod:`hybrid_source` instead.

.. deprecated::
    Import :class:`HybridGeoSource` from ``preprocessing.hybrid_source``.
"""

from .hybrid_source import HybridGeoSource as RParityGeoSource  # noqa: F401

__all__ = ["RParityGeoSource"]
