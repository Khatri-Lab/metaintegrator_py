"""
Backward-compatibility alias: use :mod:`hybrid_handler` instead.

.. deprecated::
    Import :class:`HybridMicroarrayHandler` from ``preprocessing.hybrid_handler``.
"""

from .hybrid_handler import HybridMicroarrayHandler as RParityMicroarrayHandler  # noqa: F401

__all__ = ["RParityMicroarrayHandler"]
