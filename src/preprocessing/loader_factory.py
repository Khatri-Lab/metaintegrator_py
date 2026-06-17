"""
Factory for creating pre-configured DataLoaders.

Provides a convenient, high-level API for creating loaders with
appropriate sources and handlers for common use cases.

All loader creation goes through :class:`DataLoaderFactory`.  The
convenience functions at module level are thin wrappers.

Examples
--------
# R-parity (recommended — matches R MetaIntegrator output)
>>> loader = DataLoaderFactory.create_geo_loader(strategy='r_parity')
>>> datasets = loader.load_multiple(['GSE11909', 'GSE50635'])

# Or via convenience functions:
>>> datasets = load_geo_datasets_processed(['GSE11909'], strategy='r_parity')
>>> datasets = load_geo_datasets(['GSE11909'], strategy='r_parity')  # legacy dict format
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .base import DataLoader, DataSource, DataHandler, ProcessedDataset
from .geo_sources import create_geo_source
from .microarray_handler import MicroarrayHandler
from .hybrid_source import HybridGeoSource
from .hybrid_handler import HybridMicroarrayHandler

log = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating pre-configured DataLoaders.

    This is the single canonical place for wiring sources to handlers.
    All convenience functions and thin wrappers delegate here.

    Supported GEO strategies:

    ============  =============================================  ==============
    Strategy      Description                                    Class pairing
    ============  =============================================  ==============
    ``hybrid``    Series Matrix expr + GEOparse annotation.      ``HybridGeoSource`` + ``HybridMicroarrayHandler``
                  **Recommended for production.**
    ``full``      Full SOFT download via GEOparse (all samples). ``GEOparseSource`` + ``MicroarrayHandler``
    ``fast``      Series Matrix only (may miss samples).         ``NCBIGeoSource`` + ``MicroarrayHandler``
    ============  =============================================  ==============

    ``'r_parity'`` is accepted as an alias for ``'hybrid'``.
    """

    @staticmethod
    def create_geo_loader(
        strategy: str = "full",
        cache_dir: str | Path = "~/.geo_cache",
        log_transform: bool = False,
        normalize: bool = False,
        **kwargs,
    ) -> DataLoader:
        """
        Create a DataLoader for GEO microarray data.

        Parameters
        ----------
        strategy : str, default='full'
            ``'hybrid'``: Series Matrix expr + GEOparse annotation.
                **Recommended for production.**
            ``'r_parity'``: Alias for ``'hybrid'``.
            ``'full'``: Complete dataset with ALL samples (GEOparse/SOFT).
            ``'fast'``: Subset via Series Matrix (may miss samples).
        cache_dir : str or Path, default='~/.geo_cache'
            Cache directory for downloaded files.
        log_transform : bool, default=False
            Apply log2 transformation (``full`` / ``fast`` only).
        normalize : bool, default=False
            Apply quantile normalization (``full`` / ``fast`` only).
        **kwargs
            Additional arguments passed to source constructor.

        Returns
        -------
        DataLoader
            Configured loader ready to fetch and process data.

        Examples
        --------
        >>> loader = DataLoaderFactory.create_geo_loader('hybrid')
        >>> datasets = loader.load_multiple(['GSE11909', 'GSE50635'])
        """
        if strategy in ("hybrid", "r_parity"):
            source = HybridGeoSource(cache_dir=cache_dir)
            handler = HybridMicroarrayHandler()
            return DataLoader(source, handler)

        # Non-r_parity strategies
        source = create_geo_source(strategy, cache_dir, **kwargs)
        handler = MicroarrayHandler(
            log_transform=log_transform,
            normalize=normalize,
        )
        return DataLoader(source, handler)

    @staticmethod
    def create_microarray_loader(
        source_type: str = "geo_full", cache_dir: str | Path = "~/.data_cache", **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for microarray data from various sources.

        A more general version of :meth:`create_geo_loader` that can
        be extended to support additional microarray sources.

        Parameters
        ----------
        source_type : str, default='geo_full'
            ``'geo_hybrid'``: Hybrid strategy (recommended).
            ``'geo_r_parity'``: Alias for ``'geo_hybrid'``.
            ``'geo_full'``: NCBI GEO via SOFT (complete).
            ``'geo_fast'``: NCBI GEO via Series Matrix (subset).
            ``'geo'``: Alias for ``'geo_full'``.
        cache_dir : str or Path
            Cache directory.
        **kwargs
            Additional arguments for source/handler.

        Returns
        -------
        DataLoader
        """
        source_map = {
            "geo_hybrid": "hybrid",
            "geo_r_parity": "hybrid",
            "geo_full": "full",
            "geo_fast": "fast",
            "geo": "full",
        }

        if source_type not in source_map:
            raise ValueError(
                f"Unknown source type: {source_type}. "
                f"Options: {list(source_map.keys())}"
            )

        return DataLoaderFactory.create_geo_loader(
            strategy=source_map[source_type],
            cache_dir=cache_dir,
            **kwargs,
        )

    # ========================================================================
    # Future Extensions (Placeholders)
    # ========================================================================

    @staticmethod
    def create_rnaseq_loader(
        source: str = "gdc", cache_dir: str | Path = "~/.data_cache", **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for RNA-seq data.

        Notes
        -----
        Placeholder — not yet implemented.
        """
        raise NotImplementedError(
            "RNA-seq loader not yet implemented. "
            "Coming in Phase 2 of architecture rollout."
        )

    @staticmethod
    def create_single_cell_loader(
        source: str = "cellxgene", cache_dir: str | Path = "~/.data_cache", **kwargs
    ) -> DataLoader:
        """
        Create a DataLoader for single-cell RNA-seq data.

        Notes
        -----
        Placeholder — not yet implemented.
        """
        raise NotImplementedError(
            "Single-cell loader not yet implemented. "
            "Coming in Phase 3 of architecture rollout."
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def load_geo_datasets_processed(
    gse_ids: List[str],
    strategy: str = "hybrid",
    cache_dir: str | Path = "~/.geo_cache",
    stop_on_error: bool = False,
    **kwargs,
) -> List[ProcessedDataset]:
    """
    Load GEO datasets and return :class:`ProcessedDataset` objects.

    This is the recommended entry point for new code.  It supports
    all strategies and returns typed objects.

    Parameters
    ----------
    gse_ids : list[str]
        GEO Series IDs to load.
    strategy : str, default='hybrid'
        Loader strategy (see :meth:`DataLoaderFactory.create_geo_loader`).
    cache_dir : str or Path
        Cache directory.
    stop_on_error : bool, default=False
        If ``True``, raise on first failure.
    **kwargs
        Additional arguments passed to the factory.

    Returns
    -------
    list[ProcessedDataset]
        Processed datasets ready for meta-analysis.

    Examples
    --------
    >>> datasets = load_geo_datasets_processed(['GSE11909', 'GSE50635'])
    >>> for ds in datasets:
    ...     print(f"{ds.geo_id}: {ds.expr.shape}")
    GSE11909_GPL96: (22283, 118)
    """
    loader = DataLoaderFactory.create_geo_loader(
        strategy=strategy,
        cache_dir=cache_dir,
        **kwargs,
    )
    datasets = loader.load_multiple(gse_ids, stop_on_error=stop_on_error)
    log.info(f"✓ Loaded {len(datasets)} datasets from {len(gse_ids)} GSE IDs")
    return datasets


def load_geo_datasets(
    gse_ids: list[str],
    strategy: str = "full",
    cache_dir: str = "~/.geo_cache",
    **kwargs,
) -> list[dict]:
    """
    Load GEO datasets and return legacy dict format.

    .. deprecated::
        Prefer :func:`load_geo_datasets_processed` which returns typed
        :class:`ProcessedDataset` objects and defaults to ``r_parity``.

    Parameters
    ----------
    gse_ids : list[str]
        GEO Series IDs to load.
    strategy : str, default='full'
        ``'r_parity'``, ``'full'``, or ``'fast'``.
    cache_dir : str
        Cache directory.
    **kwargs
        Additional arguments for the factory.

    Returns
    -------
    list[dict]
        Each dict has keys: ``geo_id``, ``expr``, ``pheno``,
        ``keys``, ``metadata``.
    """
    loader = DataLoaderFactory.create_geo_loader(
        strategy=strategy,
        cache_dir=cache_dir,
        **kwargs,
    )
    processed_datasets = loader.load_multiple(gse_ids)
    return [_dataset_to_dict(ds) for ds in processed_datasets]


def _dataset_to_dict(dataset: ProcessedDataset) -> dict:
    """Convert ProcessedDataset to dict for backward compatibility."""
    return {
        "geo_id": dataset.geo_id,
        "expr": dataset.expr,
        "pheno": dataset.pheno,
        "keys": dataset.keys,
        "metadata": dataset.metadata,
    }
