"""
Abstract base classes for data loading architecture.

This module defines the core abstractions for a flexible, extensible
data loading system that separates concerns:
- DataSource: WHERE data comes from (GEO, GDC, ArrayExpress, etc.)
- DataHandler: WHAT type of data and HOW to process it (microarray, RNA-seq, etc.)
- DataLoader: Orchestration layer that combines source + handler

This architecture supports:
- Multiple data sources (GEO via Series Matrix, GEO via SOFT, GDC, etc.)
- Multiple data types (microarray, RNA-seq, single-cell, proteomics)
- Easy extensibility without modifying existing code (Open/Closed Principle)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

# ============================================================================
# Data Models
# ============================================================================


@dataclass
class RawDataset:
    """
    Raw data fetched from a source before processing.

    This is the intermediate format between fetching and parsing.
    Different sources may provide different raw formats.
    """

    source_id: str  # e.g., 'GSE11909'
    platform_id: Optional[str] = None  # e.g., 'GPL96'
    raw_data: Any = None  # Source-specific raw data
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        """Default ``metadata`` to an empty dict when not provided."""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessedDataset:
    """
    Processed dataset in standard metaIntegrator format.

    This is the final output format used by analysis workflows.
    All handlers must produce this format.

    Attributes:
        geo_id: Dataset identifier (e.g. ``'GSE11909_GPL96'``).
        expr: Expression matrix, shape ``(n_features, n_samples)`` with
            probe IDs as the index and sample IDs as the columns.
        pheno: Phenotype data, shape ``(n_samples, n_metadata_cols)``,
            indexed by sample ID (must match ``expr.columns``).
        keys: Gene symbols aligned 1:1 with ``expr.index``. Multi-gene
            probes are comma-separated (e.g. ``'CD6,IL2RG'``); unmappable
            probes are ``'NA'``. This mirrors R MetaIntegrator's ``keys``,
            a character vector the same length as the number of probes.
        metadata: Additional metadata (source, platform, handler, etc.).
    """

    geo_id: str
    expr: pd.DataFrame
    pheno: pd.DataFrame
    keys: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate that expr, pheno, and keys are mutually consistent."""
        # Ensure expr and pheno have matching sample IDs
        if not self.expr.columns.equals(self.pheno.index):
            raise ValueError(
                f"Sample IDs mismatch: expr has {len(self.expr.columns)} samples, "
                f"pheno has {len(self.pheno.index)} samples"
            )
        # Ensure keys is a list aligned with expr.index
        if not isinstance(self.keys, list):
            raise TypeError(
                f"keys must be list[str], got {type(self.keys).__name__}. "
                f"If you have a dict, convert with: "
                f"list(mapping.get(p, 'NA') for p in expr.index)"
            )
        if len(self.keys) != len(self.expr):
            raise ValueError(
                f"keys length ({len(self.keys)}) must match " f"expr row count ({len(self.expr)})"
            )


# ============================================================================
# Abstract Base Classes
# ============================================================================


class DataSource(abc.ABC):
    """
    Abstract base class for data sources.

    A DataSource knows WHERE to fetch data from and HOW to retrieve it,
    but doesn't know HOW to process the data (that's the handler's job).

    Examples:
    - NCBIGeoSource: Fetches from NCBI GEO via Series Matrix
    - GEOparseSource: Fetches from NCBI GEO via SOFT files
    - GDCSource: Fetches from NCI GDC portal
    - ArrayExpressSource: Fetches from EBI ArrayExpress
    """

    def __init__(self, cache_dir: str | Path = "~/.data_cache"):
        """Initialize the data source.

        Args:
            cache_dir: Directory used to cache downloaded files.
        """
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def fetch(self, dataset_id: str, **kwargs) -> List[RawDataset]:
        """Fetch raw data from the source.

        Args:
            dataset_id: Identifier for the dataset (format depends on the
                source implementation).
            **kwargs: Source-specific parameters.

        Returns:
            One raw dataset per platform/batch found for ``dataset_id``.

        Raises:
            ValueError: If the dataset cannot be found.
            RuntimeError: If the download or fetch fails.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_multiple(
        self, dataset_ids: List[str], stop_on_error: bool = False, **kwargs
    ) -> List[RawDataset]:
        """Fetch multiple datasets.

        Args:
            dataset_ids: Dataset identifiers to fetch.
            stop_on_error: If True, raise on the first error; if False, skip
                the failed dataset and continue with the rest.
            **kwargs: Source-specific parameters.

        Returns:
            All raw datasets collected across every ID.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def source_name(self) -> str:
        """Human-readable name of this data source."""
        raise NotImplementedError


class DataHandler(abc.ABC):
    """
    Abstract base class for data type handlers.

    A DataHandler knows WHAT type of data it's processing and HOW to
    convert it to the standard metaIntegrator format.

    Examples:
    - MicroarrayHandler: Processes microarray data
    - RNASeqHandler: Processes bulk RNA-seq data
    - SingleCellHandler: Processes single-cell RNA-seq data
    - ProteomicsHandler: Processes proteomics data
    """

    @abc.abstractmethod
    def parse(self, raw_dataset: RawDataset) -> ProcessedDataset:
        """Parse raw data into the standard dataset format.

        Args:
            raw_dataset: Raw data returned by a source.

        Returns:
            The dataset in standard MetaIntegrator format.

        Raises:
            ValueError: If the raw data is invalid or cannot be parsed.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self, dataset: ProcessedDataset) -> ProcessedDataset:
        """Apply data-type-specific normalization.

        Args:
            dataset: A parsed dataset.

        Returns:
            The normalized dataset.
        """
        raise NotImplementedError

    def process(self, raw_dataset: RawDataset) -> ProcessedDataset:
        """Run the full processing pipeline: parse, then normalize.

        This is a template method — it defines the processing workflow while
        subclasses supply the actual ``parse()`` and ``normalize()`` logic.

        Args:
            raw_dataset: Raw data from a source.

        Returns:
            A fully processed dataset, ready for analysis.
        """
        parsed = self.parse(raw_dataset)
        normalized = self.normalize(parsed)
        return normalized

    @property
    @abc.abstractmethod
    def data_type(self) -> str:
        """Human-readable name of the data type this handler processes."""
        raise NotImplementedError


# ============================================================================
# Orchestration Layer
# ============================================================================


class DataLoader:
    """
    Orchestration layer that combines a DataSource and DataHandler.

    This is the main entry point for loading datasets. It coordinates
    fetching from a source and processing with a handler.

    Examples:
        >>> source = GEOparseSource()
        >>> handler = MicroarrayHandler()
        >>> loader = DataLoader(source, handler)
        >>> datasets = loader.load_multiple(['GSE11909', 'GSE50635'])
    """

    def __init__(self, source: DataSource, handler: DataHandler):
        """Initialize the loader with a source and a handler.

        Args:
            source: Where to fetch data from.
            handler: How to process the fetched data.
        """
        self.source = source
        self.handler = handler

    def load(self, dataset_id: str, **kwargs) -> List[ProcessedDataset]:
        """Load and process a single dataset.

        Args:
            dataset_id: Dataset identifier.
            **kwargs: Forwarded to ``source.fetch()``.

        Returns:
            The processed dataset(s) — more than one if the source spans
            multiple platforms.
        """
        # Fetch raw data from source
        raw_datasets = self.source.fetch(dataset_id, **kwargs)

        # Process each raw dataset with handler
        processed_datasets = []
        for raw in raw_datasets:
            try:
                processed = self.handler.process(raw)
                processed_datasets.append(processed)
            except Exception as e:
                raise RuntimeError(f"Failed to process {raw.source_id}: {e}") from e

        return processed_datasets

    def load_multiple(
        self, dataset_ids: List[str], stop_on_error: bool = False, **kwargs
    ) -> List[ProcessedDataset]:
        """Load and process multiple datasets.

        Args:
            dataset_ids: Dataset identifiers to load.
            stop_on_error: If True, raise on the first error; if False, log
                failures and return whatever succeeded.
            **kwargs: Forwarded to ``source.fetch_multiple()``.

        Returns:
            All successfully processed datasets.
        """
        # Fetch all raw datasets
        raw_datasets = self.source.fetch_multiple(
            dataset_ids, stop_on_error=stop_on_error, **kwargs
        )

        # Process each raw dataset
        processed_datasets = []
        failed = []

        for raw in raw_datasets:
            try:
                processed = self.handler.process(raw)
                processed_datasets.append(processed)
            except Exception as e:
                # Create full dataset identifier for better error messages
                dataset_id = (
                    f"{raw.source_id}_{raw.platform_id}" if raw.platform_id else raw.source_id
                )
                failed.append((dataset_id, str(e)))
                if stop_on_error:
                    raise

        if failed:
            import logging

            log = logging.getLogger(__name__)
            log.warning(f"Failed to process {len(failed)} datasets: " f"{[f[0] for f in failed]}")

        return processed_datasets

    def __repr__(self) -> str:
        return f"DataLoader(source={self.source.source_name}, " f"handler={self.handler.data_type})"
