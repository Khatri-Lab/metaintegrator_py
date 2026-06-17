# src/preprocessing/__init__.py

# Factory + convenience functions (recommended entry points)
from .loader_factory import (
    DataLoaderFactory,
    load_geo_datasets,
    load_geo_datasets_processed,
)

# Base classes
from .base import (
    DataSource,
    DataHandler,
    DataLoader,
    RawDataset,
    ProcessedDataset,
)

# GEO source implementations
from .geo_sources import (
    GEOparseSource,
    NCBIGeoSource,
    create_geo_source,
)

# Handlers and annotators
from .microarray_handler import MicroarrayHandler
from .gene_annotator import GeneAnnotator

# Network I/O
from .ncbi_network import NCBINetwork

# Hybrid architecture (recommended for production)
from .hybrid_source import HybridGeoSource
from .hybrid_handler import HybridMicroarrayHandler

# Backward-compatible aliases (deprecated — use Hybrid* instead)
from .r_parity_source import RParityGeoSource  # alias for HybridGeoSource
from .r_parity_handler import (
    RParityMicroarrayHandler,
)  # alias for HybridMicroarrayHandler
from .r_parity_loader import load_geo_datasets_r_parity  # alias, uses strategy='hybrid'

__all__ = [
    # High-level API
    "DataLoaderFactory",
    "load_geo_datasets",
    "load_geo_datasets_processed",
    "load_geo_datasets_r_parity",
    # Base classes
    "DataSource",
    "DataHandler",
    "DataLoader",
    # Implementations
    "GEOparseSource",
    "NCBIGeoSource",
    "MicroarrayHandler",
    "GeneAnnotator",
    # Hybrid (production)
    "HybridGeoSource",
    "HybridMicroarrayHandler",
    # Deprecated aliases
    "RParityGeoSource",
    "RParityMicroarrayHandler",
    # Network I/O
    "NCBINetwork",
]
