"""
Configuration schema, validation, and template generation for the
MetaIntegrator meta-analysis CLI runner.

This module defines the canonical shape of a runner config, validates
user-supplied configs against it, fills in defaults, and can emit a
fully-documented JSON or YAML template that scientists can customize.

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
    Validate and normalize a user config dict.  Returns ``(config, errors)``.
``generate_template(fmt)``
    Return a string containing a documented example config.
``DEFAULTS``
    The default values dict (read-only reference).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "validate_config",
    "generate_template",
    "load_config_file",
    "resolve_presence_threshold",
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
        # Presence-threshold gene inclusion: an opt-in, alternative way to
        # express "a gene must be measured in at least this many datasets
        # to be included" as a *percentage* of the run's dataset count,
        # instead of (or as well as) the plain integer count above.
        # `presence_threshold_value` is `None` by default, meaning "not
        # set" -- filtering then falls back entirely to
        # `number_studies_thresh`, so every config written before this
        # feature existed behaves exactly as it always has. When
        # `presence_threshold_value` *is* set, it is resolved against the
        # actual number of datasets that made it into the meta-analysis
        # (post multi-platform expansion) via
        # `resolve_presence_threshold()` below, and *that* resolved
        # integer is what actually gets passed to `MetaAnalyzer.
        # filter_genes(number_studies_thresh=...)`. See
        # `meta_analysis_runner.py::_resolve_filter_presence_threshold`
        # for the stage that performs this resolution.
        "presence_threshold_mode": "count",  # "count" | "percent"
        "presence_threshold_value": None,  # None = use number_studies_thresh as-is
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

    nst = filt.get("number_studies_thresh", 1)
    if not isinstance(nst, int) or isinstance(nst, bool) or nst < 1:
        errors.append(f"filter.number_studies_thresh must be an integer >= 1 (got {nst}).")

    presence_mode = filt.get("presence_threshold_mode", "count")
    if presence_mode not in ("count", "percent"):
        errors.append(
            f"filter.presence_threshold_mode must be 'count' or 'percent' "
            f"(got '{presence_mode}')."
        )
    presence_value = filt.get("presence_threshold_value")
    if presence_value is not None:
        if not isinstance(presence_value, (int, float)) or isinstance(presence_value, bool):
            errors.append(
                f"filter.presence_threshold_value must be a number (got {presence_value!r})."
            )
        elif presence_mode == "percent" and not (0 < presence_value <= 100):
            errors.append(
                f"filter.presence_threshold_value must be in (0, 100] when "
                f"presence_threshold_mode is 'percent' (got {presence_value})."
            )
        elif presence_mode == "count" and presence_value < 1:
            errors.append(
                f"filter.presence_threshold_value must be >= 1 when "
                f"presence_threshold_mode is 'count' (got {presence_value})."
            )

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
# Presence-threshold resolution
# =====================================================================


def resolve_presence_threshold(
    *,
    mode: str,
    value: float,
    n_datasets: int,
) -> int:
    """Resolve a gene-presence threshold to an absolute dataset count.

    A gene "present" in a dataset means the meta-analysis engine
    computed a (non-NaN) effect size for it in that dataset -- exactly
    what ``MetaAnalyzer.filter_genes``'s ``number_studies_thresh``
    parameter already filters on via the pooled results' ``numStudies``
    column. This function is the single place that turns a user-facing
    threshold choice -- either a plain count ("must appear in at least 3
    datasets") or a percentage ("must appear in at least 75% of
    datasets") -- into the absolute integer that parameter expects.

    Percentages are rounded **up** (``math.ceil``): a gene must actually
    clear the requested percentage rather than falling just short of it.
    For example, 75% of 3 datasets is 2.25, which resolves to 3 (not 2)
    -- consistent with the exact case of 4 datasets at a 75% threshold
    resolving to exactly 3, a rule that this rounding generalizes to the
    non-exact cases too.

    Args:
        mode: ``"count"`` (``value`` is an absolute minimum number of
            datasets) or ``"percent"`` (``value`` is a percentage of
            ``n_datasets``, in ``(0, 100]``).
        value: The threshold value, interpreted per ``mode``.
        n_datasets: The actual number of datasets contributing to this
            meta-analysis run (e.g. ``len(meta_object["originalData"])``
            -- the count *after* any multi-platform expansion, matching
            what ``numStudies`` is computed against).

    Returns:
        int
            The resolved absolute threshold, clamped to
            ``[1, max(1, n_datasets)]`` so it is always a satisfiable
            minimum (never demands more datasets than the run actually
            has, and never drops below "present in at least 1").

    Raises:
        ValueError: If ``mode`` is not ``"count"``/``"percent"``, or if
            ``value`` is out of range for the given ``mode``.

    Examples::

        >>> resolve_presence_threshold(mode="count", value=3, n_datasets=4)
        3
        >>> resolve_presence_threshold(mode="percent", value=75, n_datasets=4)
        3
        >>> resolve_presence_threshold(mode="percent", value=75, n_datasets=3)
        3
        >>> resolve_presence_threshold(mode="count", value=99, n_datasets=4)
        4
    """
    if mode not in ("count", "percent"):
        raise ValueError(f"mode must be 'count' or 'percent' (got {mode!r}).")

    if n_datasets < 1:
        # Degenerate: nothing to compute a threshold against (e.g. a
        # dry-run or an empty meta_object). "Present in at least 1
        # dataset" is the only meaningful bound in that case.
        return 1

    if mode == "count":
        if value < 1:
            raise ValueError(f"value must be >= 1 in 'count' mode (got {value!r}).")
        resolved = int(math.ceil(value))
    else:  # mode == "percent"
        if not (0 < value <= 100):
            raise ValueError(f"value must be in (0, 100] in 'percent' mode (got {value!r}).")
        resolved = int(math.ceil((value / 100.0) * n_datasets))

    return max(1, min(resolved, n_datasets))


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

    Raises:
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
        f"Unrecognized config file extension '{suffix}'.  " f"Use .json, .yaml, or .yml."
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
      # Presence threshold: alternative to number_studies_thresh
      # above, expressed as a percentage of this run's dataset count
      # instead of a plain count. Leave presence_threshold_value unset
      # (null) to use number_studies_thresh as-is (the default).
      # presence_threshold_mode: percent   # "count" | "percent"
      # presence_threshold_value: 75       # e.g. 75 = must appear in >=75% of datasets

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
