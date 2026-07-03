"""
Configuration schema, validation, and template generation for the
MetaIntegrator meta-analysis CLI runner.

This module defines the canonical shape of a runner config, validates
user-supplied configs against it, fills in defaults, and can emit a
fully-documented JSON or YAML template that scientists can customise.

Design decisions
----------------
* The schema is expressed as plain Python dicts rather than a third-party
  schema library so that the package has **zero extra dependencies**.
* Every field has a default so that the *minimal* valid config is just::

      { "datasets": { "gse_ids": ["GSE11909"] },
        "labeling": { "GSE11909": { "disease_terms": ["SLE"] } } }

  Everything else falls through to sensible defaults.
* The ``generate_template()`` helper emits a ready-to-edit config with
  inline comments (YAML) or an equivalent JSON file.

Public API
----------
``validate_config(raw)``
    Validate and normalise a user config dict.  Returns ``(config, errors)``.
``generate_template(fmt)``
    Return a string containing a documented example config.
``DEFAULTS``
    The default values dict (read-only reference).
"""

from __future__ import annotations

import copy
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "validate_config",
    "generate_template",
    "load_config_file",
    "DEFAULTS",
]


# =====================================================================
# Default configuration
# =====================================================================

DEFAULTS: Dict[str, Any] = {
    # ---- Data acquisition ------------------------------------------------
    "datasets": {
        "gse_ids": [],  # REQUIRED — list of GEO accessions
        "strategy": "hybrid",  # "hybrid" | "geoparse" | "ncbi"
        "cache_dir": "~/.geo_cache",  # where raw downloads are cached
        "stop_on_error": False,  # abort if one GSE fails?
    },
    # ---- Sample labeling -------------------------------------------------
    #  Keyed by GSE prefix (matched via substring, same as the notebook).
    #  Each entry must have at least "disease_terms".
    "labeling": {},  # REQUIRED — at least one entry
    # ---- Meta-analysis ---------------------------------------------------
    "meta_analysis": {
        "strategy": "random_effects",  # "random_effects" | "fixed_effects"
        "run_leave_one_out": True,
        "use_old_method": False,
        "max_cores": None,  # None = auto-detect
    },
    # ---- Gene filtering --------------------------------------------------
    "filter": {
        "fdr_thresh": 0.05,
        "effect_size_thresh": 0.0,
        "number_studies_thresh": 1,
        "heterogeneity_pval_thresh": 0.0,
        "is_leave_one_out": True,
    },
    # ---- Forward / backward gene search ----------------------------------
    "gene_search": {
        "run_backward": True,
        "run_forward": True,
        "back_thresh": 0.0,
        "forward_thresh": 0.0,
    },
    # ---- Output ----------------------------------------------------------
    "output": {
        "output_dir": "./meta_analysis_output",
        "save_plot_data": True,
        "plots": [
            "forest",
            "heatmap",
            "violin",
            "roc",
            "prc",
        ],
        "forest_gene": None,  # gene symbol for forest plot
        "violin_datasets": None,  # list of dataset IDs, or None = all
        "roc_datasets": None,  # list of dataset IDs, or None = all
        "regression_dataset": None,  # dataset ID for regression plot
        "regression_column": None,  # pheno column for regression
        "plot_dpi": 300,
        "plot_format": "png",  # "png" | "pdf" | "svg"
    },
}

# Columns that ``find_classification_column`` will try, in order,
# when the user does not supply ``possible_columns`` for a labeling entry.
_DEFAULT_POSSIBLE_COLUMNS: List[str] = [
    "characteristics",
    "title",
    "Sample_title",
    "subject type:ch1",
    "disease state:ch1",
    "characteristics_ch1",
]


# =====================================================================
# Validation
# =====================================================================


def validate_config(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Validate a user-supplied config and merge it with defaults.

    Args:
    raw : dict
        The parsed JSON / YAML content provided by the user.

    Returns:
    config : dict
        The validated and default-filled configuration.
    errors : list[str]
        A list of human-readable error messages.  Empty if valid.
    """
    errors: List[str] = []
    config = _deep_merge(copy.deepcopy(DEFAULTS), raw)

    # ---- datasets --------------------------------------------------------
    ds = config.get("datasets", {})
    gse_ids = ds.get("gse_ids", [])
    if not gse_ids or not isinstance(gse_ids, list):
        errors.append(
            "datasets.gse_ids is required and must be a non-empty list "
            'of GEO accession strings (e.g. ["GSE11909", "GSE50635"]).'
        )
    else:
        for gid in gse_ids:
            if not isinstance(gid, str) or not gid.startswith("GSE"):
                errors.append(
                    f"datasets.gse_ids: '{gid}' does not look like a valid "
                    f"GEO accession (expected 'GSExxxxx')."
                )

    strategy = ds.get("strategy", "hybrid")
    if strategy not in ("hybrid", "geoparse", "ncbi"):
        errors.append(
            f"datasets.strategy must be 'hybrid', 'geoparse', or 'ncbi' " f"(got '{strategy}')."
        )

    # ---- labeling --------------------------------------------------------
    labeling = config.get("labeling", {})
    if not labeling:
        errors.append(
            "labeling is required.  Provide at least one entry keyed by "
            "GSE prefix with 'disease_terms' (e.g. "
            '{"GSE11909": {"disease_terms": ["SLE"]}}).'
        )
    else:
        for key, entry in labeling.items():
            if not isinstance(entry, dict):
                errors.append(f"labeling.{key} must be a dict.")
                continue
            if "disease_terms" not in entry or not entry["disease_terms"]:
                errors.append(
                    f"labeling.{key}.disease_terms is required and must be "
                    f"a non-empty list of strings."
                )
            # Fill in default possible_columns if not provided
            if "possible_columns" not in entry:
                entry["possible_columns"] = list(_DEFAULT_POSSIBLE_COLUMNS)

    # ---- meta_analysis ---------------------------------------------------
    ma = config.get("meta_analysis", {})
    if ma.get("strategy", "random_effects") not in ("random_effects", "fixed_effects"):
        errors.append("meta_analysis.strategy must be 'random_effects' or 'fixed_effects'.")

    # ---- filter ----------------------------------------------------------
    filt = config.get("filter", {})
    fdr = filt.get("fdr_thresh", 0.05)
    if not isinstance(fdr, (int, float)) or not (0 < fdr <= 1):
        errors.append(f"filter.fdr_thresh must be a float in (0, 1] (got {fdr}).")

    es = filt.get("effect_size_thresh", 0.0)
    if not isinstance(es, (int, float)) or es < 0:
        errors.append(f"filter.effect_size_thresh must be >= 0 (got {es}).")

    # ---- output ----------------------------------------------------------
    out = config.get("output", {})
    valid_plots = {"forest", "heatmap", "violin", "roc", "prc", "regression"}
    requested = set(out.get("plots", []))
    unknown = requested - valid_plots
    if unknown:
        errors.append(
            f"output.plots contains unknown plot types: {sorted(unknown)}.  "
            f"Valid: {sorted(valid_plots)}."
        )

    fmt = out.get("plot_format", "png")
    if fmt not in ("png", "pdf", "svg"):
        errors.append(f"output.plot_format must be 'png', 'pdf', or 'svg' (got '{fmt}').")

    return config, errors


# =====================================================================
# Config file I/O
# =====================================================================


def load_config_file(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file.

    The format is auto-detected from the file extension (``.json``,
    ``.yaml``, ``.yml``).

    Args:
    path : str
        Path to the config file.

    Returns:
    dict
        Parsed configuration dictionary.

    Raises
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not recognized.
    """
    p = Path(path).expanduser()
    print(f"Loading configuration from {p}")

    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        return json.loads(text)

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configs.  " "Install it with: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text) or {}

    raise ValueError(
        f"Unrecognised config file extension '{suffix}'.  " f"Use .json, .yaml, or .yml."
    )


# =====================================================================
# Template generation
# =====================================================================

_TEMPLATE_YAML = textwrap.dedent("""\
    # ===========================================================================
    # MetaIntegrator CLI — Meta-Analysis Configuration
    # ===========================================================================
    # Copy this file and edit the values below.  Commented fields show their
    # defaults — uncomment and change them only when you need a non-default value.
    #
    # Run with:
    #   python -m src meta_analysis --config my_config.yaml
    #
    # Validate without running (checks config + GEO connectivity):
    #   python -m src meta_analysis --config my_config.yaml --dry-run
    # ===========================================================================

    # ---------------------------------------------------------------------------
    # 1. DATASETS — which GEO studies to download
    # ---------------------------------------------------------------------------
    datasets:
      gse_ids:
        - GSE11909          # multi-platform study (GPL96 + GPL97)
        - GSE50635
        - GSE39088
      # strategy: hybrid    # hybrid | geoparse | ncbi  (default: hybrid)
      # cache_dir: ~/.geo_cache
      # stop_on_error: false

    # ---------------------------------------------------------------------------
    # 2. LABELING — how to assign case / control labels per study
    #
    #    Keys are matched as substrings against dataset IDs, so "GSE11909" will
    #    match both GSE11909_GPL96 and GSE11909_GPL97.
    # ---------------------------------------------------------------------------
    labeling:
      GSE11909:
        disease_terms:
          - "RBP +"
          - "RBP -"
          - SLE
        possible_columns:
          - characteristics
          - characteristics_ch1
          - "disease state:ch1"

      GSE50635:
        disease_terms:
          - "RBP +"
          - "RBP -"
          - SLE
        # possible_columns uses defaults if omitted

      GSE39088:
        disease_terms:
          - SLE
          - Patient
        possible_columns:
          - characteristics
          - title
          - "disease state:ch1"
          - characteristics_ch1

    # ---------------------------------------------------------------------------
    # 3. META-ANALYSIS parameters
    # ---------------------------------------------------------------------------
    meta_analysis:
      strategy: random_effects     # random_effects | fixed_effects
      run_leave_one_out: true
      # use_old_method: false
      # max_cores: null            # null = auto-detect

    # ---------------------------------------------------------------------------
    # 4. GENE FILTERING thresholds
    # ---------------------------------------------------------------------------
    filter:
      fdr_thresh: 0.05
      effect_size_thresh: 0.0
      number_studies_thresh: 1
      heterogeneity_pval_thresh: 0.0
      is_leave_one_out: true

    # ---------------------------------------------------------------------------
    # 5. GENE SEARCH — backward and forward selection
    # ---------------------------------------------------------------------------
    gene_search:
      run_backward: true
      run_forward: true
      # back_thresh: 0.0
      # forward_thresh: 0.0

    # ---------------------------------------------------------------------------
    # 6. OUTPUT — where and what to save
    # ---------------------------------------------------------------------------
    output:
      output_dir: ./my_meta_analysis_output
      save_plot_data: true
      plot_dpi: 300
      plot_format: png             # png | pdf | svg

      # Which plots to generate (remove lines you don't need):
      plots:
        - forest
        - heatmap
        - violin
        - roc
        - prc
        # - regression

      # Plot-specific options (optional):
      # forest_gene: IFI44L         # gene to highlight in forest plot
      # violin_datasets: null        # null = all datasets
      # roc_datasets: null           # null = all datasets
      # regression_dataset: GSE50635_GPL6244
      # regression_column: age
""")


_TEMPLATE_JSON = json.dumps(
    {
        "datasets": {
            "gse_ids": ["GSE11909", "GSE50635", "GSE39088"],
            "strategy": "hybrid",
            "cache_dir": "~/.geo_cache",
            "stop_on_error": False,
        },
        "labeling": {
            "GSE11909": {
                "disease_terms": ["RBP +", "RBP -", "SLE"],
                "possible_columns": [
                    "characteristics",
                    "characteristics_ch1",
                    "disease state:ch1",
                ],
            },
            "GSE50635": {
                "disease_terms": ["RBP +", "RBP -", "SLE"],
            },
            "GSE39088": {
                "disease_terms": ["SLE", "Patient"],
                "possible_columns": [
                    "characteristics",
                    "title",
                    "disease state:ch1",
                    "characteristics_ch1",
                ],
            },
        },
        "meta_analysis": {
            "strategy": "random_effects",
            "run_leave_one_out": True,
        },
        "filter": {
            "fdr_thresh": 0.05,
            "effect_size_thresh": 0.0,
            "number_studies_thresh": 1,
            "heterogeneity_pval_thresh": 0.0,
            "is_leave_one_out": True,
        },
        "gene_search": {
            "run_backward": True,
            "run_forward": True,
        },
        "output": {
            "output_dir": "./my_meta_analysis_output",
            "save_plot_data": True,
            "plots": ["forest", "heatmap", "violin", "roc", "prc"],
            "plot_dpi": 300,
            "plot_format": "png",
        },
    },
    indent=2,
)


def generate_template(fmt: str = "yaml") -> str:
    """Return a fully-documented config template as a string.

    Args:
    fmt : str
        ``"yaml"`` (default, includes comments) or ``"json"``.

    Returns:
    str
        The template content, ready to write to a file.
    """
    if fmt.lower() in ("yaml", "yml"):
        return _TEMPLATE_YAML
    if fmt.lower() == "json":
        return _TEMPLATE_JSON
    raise ValueError(f"Unknown format '{fmt}'.  Use 'yaml' or 'json'.")


# =====================================================================
# Internal helpers
# =====================================================================


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
