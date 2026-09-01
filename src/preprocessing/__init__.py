# src/preprocessing/__init__.py

# Factory + convenience functions (recommended entry points)
# Base classes
from .base import (
    DataHandler,
    DataLoader,
    DataSource,
    ProcessedDataset,
    RawDataset,
)
from .gene_annotator import GeneAnnotator

# GEO source implementations
from .geo_sources import (
    GEOparseSource,
    NCBIGeoSource,
    create_geo_source,
)
from .hybrid_handler import HybridMicroarrayHandler

# Hybrid architecture (recommended for production)
from .hybrid_source import HybridGeoSource
from .loader_factory import (
    DataLoaderFactory,
    load_geo_datasets,
    load_geo_datasets_processed,
)

# Handlers and annotators
from .microarray_handler import MicroarrayHandler

# Network I/O
from .ncbi_network import NCBINetwork
from .r_parity_handler import (  # alias for HybridMicroarrayHandler
    RParityMicroarrayHandler,
)
from .r_parity_loader import load_geo_datasets_r_parity  # alias, uses strategy='hybrid'

# Backward-compatible aliases (deprecated: use Hybrid* instead)
from .r_parity_source import RParityGeoSource  # alias for HybridGeoSource

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
