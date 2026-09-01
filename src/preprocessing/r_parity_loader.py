"""
R-Parity GEO Data Loader: thin wrapper (backward-compatible API).

This module preserves the ``load_geo_datasets_r_parity()`` function
signature used by notebooks and tests.  Internally it delegates to
:class:`DataLoaderFactory` with ``strategy='r_parity'``.

The original monolithic implementation lived here prior to the A1+A2
refactor.  See ``r_parity_source.py`` and ``r_parity_handler.py`` for
the current implementation.

Usage
-----
    from preprocessing.r_parity_loader import load_geo_datasets_r_parity
    datasets = load_geo_datasets_r_parity(["GSE11909", "GSE50635", "GSE39088"])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .base import ProcessedDataset

log = logging.getLogger(__name__)


def load_geo_datasets_r_parity(
    gse_ids: List[str],
    cache_dir: str | Path = "~/.geo_cache",
    stop_on_error: bool = False,
) -> List[ProcessedDataset]:
    """Load GEO datasets the way R's ``getGEOData()`` does.

    Expression comes from Series Matrix files (pre-aligned, the same data R
    uses), annotation from GEOparse GPL tables (the same annotation source),
    and phenotype data from GEOparse's per-sample metadata.

    This is a convenience wrapper around::

        DataLoaderFactory.create_geo_loader(strategy='hybrid')

    Args:
        gse_ids: GEO Series IDs, e.g. ``["GSE11909", "GSE50635", "GSE39088"]``.
        cache_dir: Directory for caching downloaded files.
        stop_on_error: If True, raise on the first failure.

    Returns:
        One processed dataset per platform, ready for meta-analysis.
    """
    # Import here to avoid circular dependency
    # (loader_factory imports r_parity_source/handler;
    #  this module is imported by __init__ alongside loader_factory)
    from .loader_factory import DataLoaderFactory

    loader = DataLoaderFactory.create_geo_loader(
        strategy="hybrid",
        cache_dir=cache_dir,
    )

    datasets = loader.load_multiple(
        gse_ids,
        stop_on_error=stop_on_error,
    )

    log.info(f"✓ Loaded {len(datasets)} datasets from {len(gse_ids)} GSE IDs")
    return datasets
