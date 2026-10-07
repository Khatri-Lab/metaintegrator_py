"""
Main meta-analysis functions for MetaIntegrator Python port.

This module contains:
- :class:`EffectSizeResult`: typed container for effect-size outputs
- :class:`MetaAnalysisEngine`: ABC for pluggable analysis strategies
- :class:`InverseVarianceEngine`: shared IV base for random/fixed effects
- :class:`RandomEffectsEngine`: default random-effects implementation
- :class:`FixedEffectsEngine`: fixed-effects implementation
- :class:`MetaAnalyzer`: orchestrator that delegates to an engine
- :class:`MetaAnalysisFactory`: factory for pre-configured analyzers
- :func:`run_meta_analysis`: convenience wrapper for backward compatibility
"""

import abc
import logging
import multiprocessing as mp
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .core_analysis import FisherCombination, MetaAnalysisCore
from .data_processing import DataProcessor
from .validation import DatasetValidator

logger = logging.getLogger(__name__)

__all__ = [
    "EffectSizeResult",
    "MetaAnalysisEngine",
    "InverseVarianceEngine",
    "RandomEffectsEngine",
    "FixedEffectsEngine",
    "MetaAnalyzer",
    "MetaAnalysisFactory",
    "FilterResult",
    "run_meta_analysis",
    "filter_genes",
]


# ------------------------------------------------------------------
# Data-model contract
# ------------------------------------------------------------------
@dataclass
class EffectSizeResult:
    """Typed container for the output of :meth:`MetaAnalysisEngine.compute_effect_sizes`.

    Using a dataclass instead of a plain ``dict`` means that a custom
    engine returning ``{"pooled": df}`` instead of
    ``{"pooled_estimates": df}`` will fail immediately with an
    ``AttributeError`` at the contract boundary rather than three steps
    later in :meth:`format_results`.

    Attributes:
        g: Per-study effect-size matrix (genes × studies).
        se_g: Per-study standard-error matrix (genes × studies).
        pooled_estimates: Pooled statistics DataFrame (genes × stat cols).
    """

    g: pd.DataFrame
    se_g: pd.DataFrame
    pooled_estimates: pd.DataFrame

    # -- backward-compatible dict-like access ----------------------
    # Existing code and tests use ``result["pooled_estimates"]``.
    # These helpers keep that working while callers migrate to
    # attribute access.

    def __getitem__(self, key: str) -> Any:
        """Dict-style read access to a field, by name."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-style write access to an existing field, by name."""
        if hasattr(self, key):
            object.__setattr__(self, key, value)
        else:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Whether ``key`` names one of this dataclass's fields."""
        return key in {f.name for f in fields(self)}

    def keys(self) -> set:
        """Return field names (dict-compat)."""
        return {f.name for f in fields(self)}


# ------------------------------------------------------------------
# MetaAnalysisEngine ABC
# ------------------------------------------------------------------
class MetaAnalysisEngine(abc.ABC):
    """Abstract base class for meta-analysis strategies.

    A ``MetaAnalysisEngine`` defines the *statistical algorithm* for
    combining results across studies.  It owns three pipeline steps:

    1. **compute_effect_sizes**: per-study effect sizes → pooled estimates
    2. **compute_fisher_results**: per-study p-values → Fisher combination
    3. **format_results**: merge pooled + Fisher → final output dict

    Different engines can implement different statistical approaches
    (random effects, fixed effects, Bayesian, etc.) while sharing the
    same orchestration layer (:class:`MetaAnalyzer`).

    This mirrors the ``DataHandler`` ABC in ``src/preprocessing/base.py``.
    """

    @abc.abstractmethod
    def compute_effect_sizes(self, original_data: List[Dict]) -> EffectSizeResult:
        """Compute and combine effect sizes across studies.

        Args:
            original_data: List of dataset dictionaries.

        Returns:
            :class:`EffectSizeResult` containing per-study matrices
            and pooled statistics.
        """

    @abc.abstractmethod
    def compute_fisher_results(
        self, original_data: List[Dict], use_old_method: bool
    ) -> pd.DataFrame:
        """Compute Fisher's combination of p-values.

        Args:
            original_data: List of dataset dictionaries.
            use_old_method: Whether to use the legacy Q-value method.

        Returns:
            DataFrame with Fisher combination statistics.
        """

    @abc.abstractmethod
    def format_results(
        self, combined_results: EffectSizeResult, output_fisher: pd.DataFrame
    ) -> Dict:
        """Merge effect-size and Fisher results into a final output dict.

        Args:
            combined_results: Output of :meth:`compute_effect_sizes`.
            output_fisher: Output of :meth:`compute_fisher_results`.

        Returns:
            Dict with keys ``'datasetEffectSizes'``,
            ``'datasetEffectSizeStandardErrors'``,
            ``'pooledResults'``, ``'pooled_estimates'``,
            ``'analysisDescription'``.
        """

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description of this analysis strategy."""

    # ------ optional helpers that subclasses may override ----------

    def combine_effect_sizes(
        self,
        list_of_effects: List[pd.DataFrame],
        between_method: str = "random",
        within_method: str = "fixed_iv",
    ) -> EffectSizeResult:
        """Combine per-study effect-size DataFrames (override in subclass)."""
        raise NotImplementedError


# ------------------------------------------------------------------
# InverseVarianceEngine: shared base for IV-based methods
# ------------------------------------------------------------------
class InverseVarianceEngine(MetaAnalysisEngine):
    """Shared logic for inverse-variance meta-analysis engines.

    This base class implements the full three-step pipeline using
    inverse-variance pooling.  Subclasses only need to specify:

    * :attr:`_between_method`: ``'random'`` or ``'fixed'``
    * :attr:`description`: human-readable label

    Everything else: FDR correction, result formatting, probe
    collapse, multi-study merging, Fisher adjustment: is
    method-independent and lives here.

    Args:
        core: Effect-size backend (default :class:`MetaAnalysisCore`).
        fisher: Fisher combination backend (default :class:`FisherCombination`).
    """

    def __init__(self, core: Optional[Any] = None, fisher: Optional[Any] = None) -> None:
        """Initialize the engine.

        Args:
            core: Effect-size backend. Defaults to :class:`MetaAnalysisCore`.
            fisher: Fisher combination backend. Defaults to
                :class:`FisherCombination`.
        """
        self._core = core or MetaAnalysisCore
        self._fisher = fisher or FisherCombination

    @property
    @abc.abstractmethod
    def _between_method(self) -> str:
        """Pooling method passed to ``pool_inverse_variance``.

        Returns ``'random'`` or ``'fixed'``.
        """

    # ---- pipeline step 1 ----------------------------------------

    def compute_effect_sizes(self, original_data: List[Dict]) -> EffectSizeResult:
        """Compute effect sizes, apply FDR correction, sort by significance."""
        logger.debug("Computing effect sizes across %d studies", len(original_data))

        all_effect_sizes = []
        for i, dataset in enumerate(original_data):
            es = self._core.calculate_effect_sizes(dataset)
            logger.debug(
                "Study %d: %d probes, has 'keys' column: %s%s",
                i + 1,
                len(es),
                "keys" in es.columns,
                (f", unique genes: {es['keys'].nunique()}" if "keys" in es.columns else ""),
            )
            all_effect_sizes.append(es)

        logger.debug("Combining effect sizes across %d studies...", len(all_effect_sizes))
        combined = self.combine_effect_sizes(all_effect_sizes)

        pooled_es = combined.pooled_estimates.copy()

        logger.debug("Applying FDR correction...")
        pooled_es["p_fdr"] = multipletests(pooled_es["p_value"].fillna(1), method="fdr_bh")[1]

        # Backward-compatible alias
        pooled_es["effect_size"] = pooled_es["summary"]

        # Sort matching R: order(pooledResults$effectSizeFDR, pooledResults$effectSizePval)
        pooled_es_sorted = pooled_es.sort_values(["p_fdr", "p_value"])

        logger.debug(
            "Sorted pooled estimates: %d genes before sort, %d genes after sort",
            len(pooled_es),
            len(pooled_es_sorted),
        )

        return EffectSizeResult(
            g=combined.g,
            se_g=combined.se_g,
            pooled_estimates=pooled_es_sorted,
        )

    # ---- pipeline step 2 ----------------------------------------

    def compute_fisher_results(
        self, original_data: List[Dict], use_old_method: bool
    ) -> pd.DataFrame:
        """Compute Fisher's combination of per-study p-values."""
        logger.debug("Computing Fisher's method")

        if use_old_method:
            warnings.warn("Old method with Q-values not fully implemented")
            return self._compute_fisher_old(original_data)

        all_pvalues = []
        for i, dataset in enumerate(original_data):
            pvals = self._fisher.calculate_study_pvalues(dataset)
            logger.debug(
                "Study %d: %d p-value genes, has 'keys' column: %s",
                i + 1,
                len(pvals),
                "keys" in pvals.columns,
            )
            all_pvalues.append(pvals)

        logger.debug("Combining p-values using Fisher's method...")
        output_fisher = self._fisher.sum_of_logs(all_pvalues)
        logger.debug("Fisher result genes: %d", len(output_fisher))

        return self._adjust_fisher(output_fisher, method="fdr_bh")

    # ---- pipeline step 3 ----------------------------------------

    def format_results(
        self, combined_results: EffectSizeResult, output_fisher: pd.DataFrame
    ) -> Dict:
        """Merge pooled and Fisher results into the final output dict."""
        pooled_es = combined_results.pooled_estimates

        logger.debug(
            "Formatting final results: pooled=%d genes, Fisher=%d genes",
            len(pooled_es),
            len(output_fisher),
        )

        pooled_results = pooled_es[
            [
                "summary",
                "se_summary",
                "p_value",
                "p_fdr",
                "tau2",
                "n_studies",
                "Q",
                "pval_het",
            ]
        ].copy()
        pooled_results.columns = [
            "effectSize",
            "effectSizeStandardError",
            "effectSizePval",
            "effectSizeFDR",
            "tauSquared",
            "numStudies",
            "cochranesQ",
            "heterogeneityPval",
        ]

        fisher_results = output_fisher[
            [
                "F.stat.up",
                "F.pval.up",
                "F.Qval.up",
                "F.stat.down",
                "F.pval.down",
                "F.Qval.down",
            ]
        ].copy()
        fisher_results.columns = [
            "fisherStatUp",
            "fisherPvalUp",
            "fisherFDRUp",
            "fisherStatDown",
            "fisherPvalDown",
            "fisherFDRDown",
        ]

        logger.debug(
            "Merging pooled (%d) and Fisher (%d) results (outer join)...",
            len(pooled_results),
            len(fisher_results),
        )

        final_results = pooled_results.join(fisher_results, how="outer")
        logger.debug("After merge: %d genes", len(final_results))

        # Sort matching R: order(pooledResults$effectSizeFDR, pooledResults$effectSizePval)
        final_results = final_results.sort_values(["effectSizeFDR", "effectSizePval"])

        return {
            "datasetEffectSizes": combined_results.g,
            "datasetEffectSizeStandardErrors": combined_results.se_g,
            "pooledResults": final_results,
            "pooled_estimates": combined_results.pooled_estimates,
            "analysisDescription": self.description,
        }

    # ---- combine_effect_sizes ------------------------------------

    def combine_effect_sizes(
        self,
        list_of_effects: List[pd.DataFrame],
        between_method: Optional[str] = None,
        within_method: str = "fixed_iv",
    ) -> EffectSizeResult:
        """Combine per-study effect sizes into pooled estimates.

        Collapses probes to genes, merges across studies, separates
        effect-size / SE matrices, and pools via inverse-variance.

        Args:
            list_of_effects: One DataFrame per study (columns ``g``,
                ``se_g``, ``keys``).
            between_method: ``'random'`` or ``'fixed'``.  Defaults to
                the engine's :attr:`_between_method`.
            within_method: ``'fixed_iv'`` or ``'extreme'``.

        Returns:
            :class:`EffectSizeResult` with per-study matrices and
            pooled statistics.
        """
        if not list_of_effects:
            raise ValueError("List of effects cannot be empty")

        if between_method is None:
            between_method = self._between_method

        logger.debug(
            "Combine effect sizes pipeline: %d studies, between=%s, within=%s",
            len(list_of_effects),
            between_method,
            within_method,
        )

        study_names = [f"study_{i}" for i in range(len(list_of_effects))]

        study_effects = [
            self._collapse_study(ef, within_method, i) for i, ef in enumerate(list_of_effects)
        ]

        merged_df = self._multimerge_effect_sizes(study_effects, study_names)

        if merged_df.index.duplicated().any():
            logger.warning(
                "Merged index has %d DUPLICATE GENES",
                merged_df.index.duplicated().sum(),
            )

        g_matrix, se_matrix = self._separate_matrices(merged_df)

        logger.debug("Pooling effect sizes (%s effects)", between_method)
        pooled = self._core.pool_inverse_variance(g_matrix, se_matrix, method=between_method)
        logger.debug("Pooled results: %d genes", len(pooled))

        if pooled.index.duplicated().any():
            logger.warning(
                "Pooled results have %d DUPLICATE GENES",
                pooled.index.duplicated().sum(),
            )

        return EffectSizeResult(g=g_matrix, se_g=se_matrix, pooled_estimates=pooled)

    # ---- private helpers -----------------------------------------

    def _collapse_study(
        self, effects_df: pd.DataFrame, within_method: str, study_index: int
    ) -> pd.DataFrame:
        """Collapse probes to genes for a single study."""
        logger.debug(
            "Study %d: shape=%s, columns=%s",
            study_index + 1,
            effects_df.shape,
            list(effects_df.columns),
        )

        if "keys" not in effects_df.columns:
            # Without keys there is no probe-to-gene mapping, so the study
            # cannot be collapsed to genes or merged with other studies.
            raise ValueError(
                f"Study {study_index + 1}: effect-size table has no 'keys' column. "
                "Each study needs one gene key per probe."
            )

        unique_genes = effects_df["keys"].nunique()
        duplicate_keys = effects_df["keys"].duplicated().sum()
        logger.debug(
            "Study %d: %d rows, %d unique genes, %d duplicate keys",
            study_index + 1,
            len(effects_df),
            unique_genes,
            duplicate_keys,
        )

        collapsed = self._core.collapse_probes_to_genes(effects_df, method=within_method)
        logger.debug("Study %d after collapse: shape=%s", study_index + 1, collapsed.shape)

        if collapsed.index.duplicated().any():
            logger.warning(
                "Study %d: index has %d DUPLICATE GENES after collapse",
                study_index + 1,
                collapsed.index.duplicated().sum(),
            )

        return collapsed

    @staticmethod
    def _separate_matrices(
        merged_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split merged DataFrame into (g_matrix, se_matrix)."""
        se_matrix = merged_df[[c for c in merged_df.columns if c.endswith("_se_g")]]
        g_matrix = merged_df[
            [c for c in merged_df.columns if c.endswith("_g") and not c.endswith("_se_g")]
        ]
        logger.debug(
            "Effect size matrix: %s, Standard error matrix: %s",
            g_matrix.shape,
            se_matrix.shape,
        )
        return g_matrix, se_matrix

    @staticmethod
    def _multimerge_effect_sizes(
        study_effects: List[pd.DataFrame], study_names: List[str]
    ) -> pd.DataFrame:
        """Merge per-study effect sizes via ``pd.concat`` + ``reindex``."""
        all_genes = sorted({gene for ef in study_effects for gene in ef.index})
        logger.debug("Total unique genes across studies: %d", len(all_genes))

        frames = []
        for effects, name in zip(study_effects, study_names):
            logger.debug("%s: %d genes", name, len(effects))
            frames.append(effects.rename(columns=lambda c: f"{name}_{c}"))

        return pd.concat(frames, axis=1).reindex(all_genes)

    @staticmethod
    def _adjust_fisher(fisher_output: pd.DataFrame, method: str = "fdr_bh") -> pd.DataFrame:
        """Adjust Fisher p-values for multiple testing."""
        fisher_adj = fisher_output.copy()

        if "F.pval.up" in fisher_adj.columns:
            _, adjusted_up, _, _ = multipletests(fisher_adj["F.pval.up"].fillna(1), method=method)
            fisher_adj["F.Qval.up"] = adjusted_up

        if "F.pval.down" in fisher_adj.columns:
            _, adjusted_down, _, _ = multipletests(
                fisher_adj["F.pval.down"].fillna(1), method=method
            )
            fisher_adj["F.Qval.down"] = adjusted_down

        return fisher_adj


# ------------------------------------------------------------------
# RandomEffectsEngine
# ------------------------------------------------------------------
class RandomEffectsEngine(InverseVarianceEngine):
    """Standard inverse-variance random-effects meta-analysis engine.

    This is the default engine used by :class:`MetaAnalyzer`.  It
    replicates the statistical pipeline of the R ``MetaIntegrator``
    package, using DerSimonian–Laird random-effects pooling.

    All statistical logic lives in :class:`InverseVarianceEngine`;
    this subclass only specifies ``_between_method = "random"``.

    Args:
        core: Effect-size backend (default :class:`MetaAnalysisCore`).
        fisher: Fisher combination backend (default :class:`FisherCombination`).
    """

    @property
    def _between_method(self) -> str:
        """Pooling method: ``'random'``."""
        return "random"

    @property
    def description(self) -> str:
        """Human-readable name of this engine."""
        return "MetaAnalysis: Random Effects Model"


# ------------------------------------------------------------------
# FixedEffectsEngine
# ------------------------------------------------------------------
class FixedEffectsEngine(InverseVarianceEngine):
    """Inverse-variance fixed-effects meta-analysis engine.

    Identical pipeline to :class:`RandomEffectsEngine` except that
    between-study variance (τ²) is assumed to be zero, so pooled
    estimates are weighted by within-study precision only.

    Args:
        core: Effect-size backend (default :class:`MetaAnalysisCore`).
        fisher: Fisher combination backend (default :class:`FisherCombination`).
    """

    @property
    def _between_method(self) -> str:
        """Pooling method: ``'fixed'``."""
        return "fixed"

    @property
    def description(self) -> str:
        """Human-readable name of this engine."""
        return "MetaAnalysis: Fixed Effects Model"


# ------------------------------------------------------------------
# Filter result container
# ------------------------------------------------------------------
@dataclass
class FilterResult:
    """Container for gene-filtering results.

    Mirrors the R ``filterGenes()`` return structure so that
    downstream code and notebooks can use the same field names.

    Attributes:
        pos_gene_names: Up-regulated genes passing all thresholds.
        neg_gene_names: Down-regulated genes passing all thresholds.
        n_pos: Number of positive genes.
        n_neg: Number of negative genes.
        effect_size_thresh: Effect-size threshold used.
        fdr_thresh: FDR threshold used.
        number_studies_thresh: Minimum-studies threshold used.
        heterogeneity_pval_thresh: Heterogeneity p-value threshold used.
        is_leave_one_out: Whether R's leave-one-out rule was applied.
        filter_description: Human-readable description of the filter.
    """

    pos_gene_names: List[str]
    neg_gene_names: List[str]
    n_pos: int
    n_neg: int
    effect_size_thresh: float
    fdr_thresh: float
    number_studies_thresh: int
    heterogeneity_pval_thresh: float
    is_leave_one_out: bool
    filter_description: str

    # -- dict-like access for backward compatibility ----------------
    def __getitem__(self, key: str) -> Any:
        """Allow ``result['posGeneNames']`` (R-style) or ``result['pos_gene_names']``."""
        _aliases = {
            "posGeneNames": "pos_gene_names",
            "negGeneNames": "neg_gene_names",
            "effectSizeThresh": "effect_size_thresh",
            "FDRThresh": "fdr_thresh",
            "numberStudiesThresh": "number_studies_thresh",
            "heterogeneityPvalThresh": "heterogeneity_pval_thresh",
            "isLeaveOneOut": "is_leave_one_out",
            "filterDescription": "filter_description",
        }
        attr = _aliases.get(key, key)
        try:
            return getattr(self, attr)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Whether ``key`` names a field, in either R-style or Python-style naming."""
        _aliases = {
            "posGeneNames": "pos_gene_names",
            "negGeneNames": "neg_gene_names",
        }
        attr = _aliases.get(key, key)
        return hasattr(self, attr)

    @property
    def total_genes(self) -> int:
        """Total number of filtered genes (positive + negative)."""
        return self.n_pos + self.n_neg

    # ------------------------------------------------------------
    # helper – a typed, R‑style dict conversion
    # ------------------------------------------------------------
    def to_dict(self) -> dict:
        """
        Return a plain `dict` that mimics the structure that older
        (R‑style) code expects.

        The returned dict contains *exactly* the keys that the original
        ``__getitem__`` implementation supported – the ones that
        existed in the 0.x API.  This keeps backward‑compatibility
        **without** leaking the new attribute names to callers.

        Example
        -------
        >>> fr = FilterResult(
        ...     pos_gene_names=['A', 'B'],
        ...     neg_gene_names=['C'],
        ...     n_pos=2,
        ...     n_neg=1,
        ...     effect_size_thresh=0.5,
        ...     fdr_thresh=0.01,
        ...     number_studies_thresh=3,
        ...     heterogeneity_pval_thresh=0.2,
        ...     is_leave_one_out=False,
        ...     filter_description='some filter'
        ... )
        >>> d = fr.to_dict()
        >>> d['posGeneNames']        # doctest: +ELLIPSIS
        ['A', 'B']
        >>> d['numberStudiesThresh']   # doctest: +ELLIPSIS
        3
        """
        # 1.️ Pull the internal dataclass representation into a plain dict
        raw = asdict(self)

        # 2.️ Build the backwards‑compatibility mapping – the same as in __getitem__
        alias_map = {
            "pos_gene_names": "posGeneNames",
            "neg_gene_names": "negGeneNames",
            "effect_size_thresh": "effectSizeThresh",
            "fdr_thresh": "FDRThresh",
            "number_studies_thresh": "numberStudiesThresh",
            "heterogeneity_pval_thresh": "heterogeneityPvalThresh",
            "is_leave_one_out": "isLeaveOneOut",
            "filter_description": "filterDescription",
        }

        # 3.️ Convert the keys
        result: dict[str, object] = {}
        for key, value in raw.items():
            # if the key is part of the alias map, rename it; otherwise
            # keep the original name (this should never happen in the
            # current data model, but it makes the helper future‑proof).
            result[alias_map.get(key, key)] = value

        # 4.️ Add the *derived* property `total_genes` – callers
        #      historically accessed it via `result['totalGenes']` only
        #      through the property itself, so we expose it here as well.
        result["totalGenes"] = self.total_genes

        return result


# ------------------------------------------------------------------
# Module-level leave-one-out worker (must live at module scope: see
# MetaAnalyzer.run_leave_one_out_analysis for why)
# ------------------------------------------------------------------
def _run_loo_subset(
    analyzer: "MetaAnalyzer", subset_data: List[Dict], use_old_method: bool
) -> Dict:
    """Run one leave-one-out subset's core meta-analysis.

     This is a free function at module scope, not a method or a nested
     closure, because :class:`~concurrent.futures.ProcessPoolExecutor`
     (used by :meth:`MetaAnalyzer.run_leave_one_out_analysis` for
     ``max_cores > 1``) has to pickle every callable it submits so it can be
     sent to a worker process. Pickle can only resolve a callable "by name"
    : via its fully-qualified module path: and a function defined *inside*
     another function (a closure) has no such path, so it always fails with
     ``Can't pickle local object '...run_leave_one_out_analysis.<locals>.
     analyze_subset'``. That was a real, previously-uncaught bug: every LOO
     subset raised this exception, so ``run_leave_one_out_analysis`` silently
     returned a list of four ``None`` results and downstream leave-one-out
     consistency filtering in :meth:`MetaAnalyzer.filter_genes` was a no-op
     even when ``is_leave_one_out=True`` in the config, because
     ``meta_object.get("leaveOneOutAnalysis")`` evaluated as an empty (falsy)
     dict.

     ``analyzer`` (a :class:`MetaAnalyzer` instance) is passed explicitly so
     the worker process can call the *same* engine/validator/processor the
     caller configured, rather than duplicating that wiring here. This is
     picklable for the default configuration (its collaborators are plain
     module-level classes), matching the exact scenario in the reported bug.

     Args:
         analyzer: The :class:`MetaAnalyzer` instance to run the subset on.
         subset_data: The dataset list with one dataset excluded.
         use_old_method: Use legacy Q-value Fisher method.

     Returns:
         The subset's core meta-analysis result dict (same shape as
         :meth:`MetaAnalyzer.run_meta_analysis_core`'s return value).
    """
    return analyzer.run_meta_analysis_core(subset_data, use_old_method)


# ------------------------------------------------------------------
# MetaAnalyzer: orchestrator
# ------------------------------------------------------------------
class MetaAnalyzer:
    """Main class for conducting meta-analysis.

    ``MetaAnalyzer`` is the orchestration layer.  It handles
    validation, parallelism, and leave-one-out logic, delegating the
    core statistical computation to a :class:`MetaAnalysisEngine`.

    The engine is the single point of contact for all pipeline steps.
    Callers that need to mock or replace individual steps should
    patch ``analyzer._engine.compute_effect_sizes``, etc., rather
    than patching the orchestrator itself.  The only delegate that
    remains on ``MetaAnalyzer`` is :meth:`combine_effect_sizes`,
    which is public API.

    Collaborators can be injected via the constructor for testing and
    extensibility:

    * **engine**: a :class:`MetaAnalysisEngine` (owns core + fisher)
    * **core / fisher**: shortcut: auto-wrapped into a
      :class:`RandomEffectsEngine` when *engine* is ``None``
    * **validator / processor**: dataset helpers

    Examples::

        # Default (random effects)
        analyzer = MetaAnalyzer()

        # Phase-2 DI style (backward-compatible)
        analyzer = MetaAnalyzer(core=mock_core, fisher=mock_fisher)

        # Phase-3 engine style
        analyzer = MetaAnalyzer(engine=RandomEffectsEngine())

        # Via factory
        analyzer = MetaAnalysisFactory.create_analyzer("random_effects")
    """

    def __init__(
        self,
        max_cores: Optional[int] = None,
        core: Optional[Any] = None,
        fisher: Optional[Any] = None,
        validator: Optional[Any] = None,
        processor: Optional[Any] = None,
        engine: Optional[MetaAnalysisEngine] = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            max_cores: Maximum worker processes to use for parallel
                steps. Defaults to ``min(10, cpu_count())``.
            core: Effect-size backend, passed through to the engine
                when *engine* is not given. Defaults to
                :class:`MetaAnalysisCore`.
            fisher: Fisher combination backend, passed through to the
                engine when *engine* is not given. Defaults to
                :class:`FisherCombination`.
            validator: Dataset validator. Defaults to
                :class:`DatasetValidator`.
            processor: Dataset processor. Defaults to
                :class:`DataProcessor`.
            engine: A pre-built :class:`MetaAnalysisEngine`. When given,
                *core* and *fisher* are ignored; otherwise a
                :class:`RandomEffectsEngine` is built from them.
        """
        if max_cores is None:
            max_cores = min(10, mp.cpu_count())
        else:
            max_cores = min(max_cores, 10, mp.cpu_count())

        self.max_cores = max_cores

        # Engine owns core + fisher; accept both injection styles
        if engine is not None:
            self._engine = engine
        else:
            self._engine = RandomEffectsEngine(core=core, fisher=fisher)

        self._validator = validator or DatasetValidator
        self._processor = processor or DataProcessor

    # ---- pipeline orchestration ----------------------------------

    def run_meta_analysis_core(
        self, original_data: List[Dict], use_old_method: bool = False
    ) -> Dict:
        """Run the core meta-analysis pipeline.

        Pipeline steps (delegated to the engine):
        1. Create combined annotation table
        2. Compute & pool effect sizes
        3. Compute Fisher's combination
        4. Format and merge final results

        Args:
            original_data: Non-empty list of dataset dicts.
            use_old_method: Use legacy Q-value Fisher method.

        Returns:
            Dict with keys ``'pooledResults'``, ``'pooled_estimates'``,
            ``'datasetEffectSizes'``, etc.
        """
        if not original_data:
            raise ValueError("Original data cannot be empty")

        logger.info("Creating combined annotation table...")
        self._processor.create_combined_annotation_table(original_data)

        logger.info("Computing effect sizes...")
        combined_results = self._engine.compute_effect_sizes(original_data)

        logger.debug(
            "After effect size computation: pooled estimates genes = %d",
            len(combined_results.pooled_estimates),
        )

        logger.info("Computing Fisher's output...")
        output_fisher = self._engine.compute_fisher_results(original_data, use_old_method)

        logger.debug("After Fisher computation: Fisher results genes = %d", len(output_fisher))

        final_results = self._engine.format_results(combined_results, output_fisher)

        if "pooledResults" in final_results:
            total_genes = len(final_results["pooledResults"])
            logger.info("Total genes analyzed: %d", total_genes)

        return final_results

    # ---- public delegate (part of the MetaAnalyzer API) ---------

    def combine_effect_sizes(self, *args, **kwargs) -> EffectSizeResult:
        """Delegate to engine.combine_effect_sizes.

        This is the only delegate that remains on ``MetaAnalyzer``
        because it is part of the public API.  All underscore-prefixed
        pipeline methods are accessed via ``analyzer._engine`` directly.
        """
        return self._engine.combine_effect_sizes(*args, **kwargs)

    # ---- gene filtering (R-compatible filterGenes) ---------------

    def filter_genes(
        self,
        meta_object: Dict,
        effect_size_thresh: float = 0.0,
        fdr_thresh: float = 0.05,
        number_studies_thresh: int = 1,
        heterogeneity_pval_thresh: float = 0.0,
        is_leave_one_out: bool = True,
    ) -> Dict:
        """Filter meta-analysis results to identify significant genes.

        This is the Python equivalent of R ``filterGenes()``. It stores the
        outcome under the ``'filterResults'`` key of *meta_object*.

        The core rule is a port of R's ``.filterMetaRun()``. A gene passes
        when all of these hold:

        1. ``|effectSize|`` >= *effect_size_thresh*
        2. ``effectSizeFDR`` <= *fdr_thresh*
        3. ``numStudies`` >= the studies threshold
        4. ``heterogeneityPval`` >= *heterogeneity_pval_thresh*
           (skipped when the threshold is 0)
        5. Fisher FDR <= *fdr_thresh* in the gene's direction:
           ``fisherFDRUp`` for up genes, ``fisherFDRDown`` for down genes.

        Without leave-one-out, the rule runs once on the main
        ``pooledResults`` with *number_studies_thresh*.

        With leave-one-out, R works differently, and so do we. The main
        result is **not** filtered. Instead the rule runs on every
        leave-one-out ``pooledResults``, with *number_studies_thresh* - 1
        because one study is missing. A gene is kept only if it passes in
        **every** leave-one-out run. The gene order comes from the first run.

        Leave-one-out is used only when it was requested, its results
        exist, and there are more than two datasets. Otherwise the plain
        rule runs, as in R, and ``is_leave_one_out`` in the result is
        ``False``.

        Args:
            meta_object: Dict returned by :meth:`run_meta_analysis`.
                Must contain ``'metaAnalysis'`` with ``'pooledResults'``.
            effect_size_thresh: Minimum absolute effect size (default 0).
            fdr_thresh: Maximum FDR for both effect-size and Fisher
                tests (default 0.05).
            number_studies_thresh: Minimum number of contributing
                studies (default 1).
            heterogeneity_pval_thresh: Minimum heterogeneity p-value
                (default 0 = disabled).
            is_leave_one_out: If ``True`` and leave-one-out results are
                present, use R's leave-one-out rule (default ``True``).

        Returns:
            The *meta_object* dict with an added or updated
            ``'filterResults'`` key containing a :class:`FilterResult`.

        Raises:
            ValueError: If ``meta_object`` is missing required keys or
                threshold values are out of range.

        Examples::

            analyzer = MetaAnalyzer()
            meta = analyzer.run_meta_analysis(meta_object)
            meta = analyzer.filter_genes(
                meta, effect_size_thresh=0.5, fdr_thresh=0.05
            )
            print(meta['filterResults'].pos_gene_names)
        """
        # ---- input validation ----------------------------------------
        if not 0 <= fdr_thresh <= 1:
            raise ValueError("fdr_thresh must be between 0 and 1")
        if effect_size_thresh < 0:
            raise ValueError("effect_size_thresh must be non-negative")
        if number_studies_thresh < 1:
            raise ValueError("number_studies_thresh must be at least 1")
        if not 0 <= heterogeneity_pval_thresh <= 1:
            raise ValueError("heterogeneity_pval_thresh must be between 0 and 1")

        if "metaAnalysis" not in meta_object or meta_object["metaAnalysis"] is None:
            raise ValueError(
                "meta_object must contain 'metaAnalysis'. " "Run run_meta_analysis() first."
            )
        meta_analysis = meta_object["metaAnalysis"]
        if "pooledResults" not in meta_analysis:
            raise ValueError(
                "metaAnalysis must contain 'pooledResults'. " "Run run_meta_analysis() first."
            )

        pooled: pd.DataFrame = meta_analysis["pooledResults"]
        logger.info(
            "filter_genes: %d genes, thresholds: "
            "|ES|>=%.2f, FDR<=%.4f, nStudies>=%d, hetPval>=%.2f, LOO requested=%s",
            len(pooled),
            effect_size_thresh,
            fdr_thresh,
            number_studies_thresh,
            heterogeneity_pval_thresh,
            is_leave_one_out,
        )

        loo_tables = self._usable_loo_tables(meta_object) if is_leave_one_out else []
        applied_loo = bool(loo_tables)

        if applied_loo:
            # R filterGenes(isLeaveOneOut = TRUE): filter each LOO run with
            # one fewer study required, then intersect. The main result is
            # deliberately not filtered.
            per_run = [
                self._filter_pooled_results(
                    table,
                    effect_size_thresh=effect_size_thresh,
                    fdr_thresh=fdr_thresh,
                    number_studies_thresh=number_studies_thresh - 1,
                    heterogeneity_pval_thresh=heterogeneity_pval_thresh,
                )
                for table in loo_tables
            ]
            pos_genes, neg_genes = per_run[0]
            for run_pos, run_neg in per_run[1:]:
                run_pos_set, run_neg_set = set(run_pos), set(run_neg)
                pos_genes = [g for g in pos_genes if g in run_pos_set]
                neg_genes = [g for g in neg_genes if g in run_neg_set]
            logger.debug(
                "Leave-one-out rule over %d runs: %d pos, %d neg",
                len(loo_tables),
                len(pos_genes),
                len(neg_genes),
            )
        else:
            pos_genes, neg_genes = self._filter_pooled_results(
                pooled,
                effect_size_thresh=effect_size_thresh,
                fdr_thresh=fdr_thresh,
                number_studies_thresh=number_studies_thresh,
                heterogeneity_pval_thresh=heterogeneity_pval_thresh,
            )

        # ---- build result --------------------------------------------
        desc = (
            f"filter_genes: |ES|≥{effect_size_thresh}, "
            f"FDR≤{fdr_thresh}, nStudies≥{number_studies_thresh}, "
            f"hetPval≥{heterogeneity_pval_thresh}, LOO={applied_loo}"
        )
        result = FilterResult(
            pos_gene_names=pos_genes,
            neg_gene_names=neg_genes,
            n_pos=len(pos_genes),
            n_neg=len(neg_genes),
            effect_size_thresh=effect_size_thresh,
            fdr_thresh=fdr_thresh,
            number_studies_thresh=number_studies_thresh,
            heterogeneity_pval_thresh=heterogeneity_pval_thresh,
            is_leave_one_out=applied_loo,
            filter_description=desc,
        )

        meta_object["filterResults"] = result

        logger.info(
            "filter_genes complete: %d positive, %d negative genes",
            result.n_pos,
            result.n_neg,
        )
        return meta_object

    # ---- gene-filter helpers ---------------------------------------

    @staticmethod
    def _filter_pooled_results(
        pooled: pd.DataFrame,
        effect_size_thresh: float,
        fdr_thresh: float,
        number_studies_thresh: int,
        heterogeneity_pval_thresh: float,
    ) -> Tuple[List[str], List[str]]:
        """Apply R's ``.filterMetaRun()`` rule to one ``pooledResults`` table.

        Comparisons with NaN are False, so genes with missing values drop
        out. R's ``which()`` does the same. Optional columns
        (``numStudies``, ``heterogeneityPval`` and the Fisher FDRs) are
        skipped with a debug message when absent. R would fail instead,
        but older or hand-built tables may lack them.

        Args:
            pooled: A ``pooledResults`` table indexed by gene.
            effect_size_thresh: Minimum absolute effect size.
            fdr_thresh: Maximum effect-size FDR and Fisher FDR.
            number_studies_thresh: Minimum number of studies. May be 0 in
                leave-one-out runs.
            heterogeneity_pval_thresh: Minimum heterogeneity p-value
                (0 disables this filter).

        Returns:
            ``(pos_gene_names, neg_gene_names)`` in table order.
        """
        res = pooled[pooled["effectSize"].abs() >= effect_size_thresh]
        res = res[res["effectSizeFDR"] <= fdr_thresh]

        if "numStudies" in res.columns:
            res = res[res["numStudies"] >= number_studies_thresh]
        else:
            logger.debug("numStudies column absent; skipping studies filter")

        if heterogeneity_pval_thresh > 0:
            if "heterogeneityPval" in res.columns:
                res = res[res["heterogeneityPval"] >= heterogeneity_pval_thresh]
            else:
                logger.debug("heterogeneityPval column absent; skipping heterogeneity filter")

        pos = res[res["effectSize"] > 0]
        neg = res[res["effectSize"] < 0]
        if "fisherFDRUp" in pos.columns:
            pos = pos[pos["fisherFDRUp"] <= fdr_thresh]
        else:
            logger.debug("fisherFDRUp column absent; skipping Fisher filter for pos genes")
        if "fisherFDRDown" in neg.columns:
            neg = neg[neg["fisherFDRDown"] <= fdr_thresh]
        else:
            logger.debug("fisherFDRDown column absent; skipping Fisher filter for neg genes")

        return pos.index.tolist(), neg.index.tolist()

    @staticmethod
    def _usable_loo_tables(meta_object: Dict) -> List[pd.DataFrame]:
        """Return the leave-one-out tables that R's rule can use.

        R applies its leave-one-out rule only when leave-one-out results
        exist and there are more than two datasets. A sub-analysis that
        failed (``None``) is left out with a warning, because the rule can
        then only use the runs that succeeded.

        Args:
            meta_object: Dict that may hold ``leaveOneOutAnalysis`` and
                ``originalData``.

        Returns:
            The usable ``pooledResults`` tables. An empty list means "use
            the plain rule instead".
        """
        loo = meta_object.get("leaveOneOutAnalysis") or {}
        original_data = meta_object.get("originalData")
        if not loo or (original_data is not None and len(original_data) <= 2):
            return []

        tables = [
            result["pooledResults"]
            for result in loo.values()
            if result is not None and result.get("pooledResults") is not None
        ]
        n_missing = len(loo) - len(tables)
        if tables and n_missing:
            logger.warning(
                "%d of %d leave-one-out runs have no results. The leave-one-out "
                "filter uses only the %d runs that succeeded.",
                n_missing,
                len(loo),
                len(tables),
            )
        elif not tables:
            logger.warning("All leave-one-out runs are empty. Falling back to the plain filter.")
        return tables

    # ---- leave-one-out -------------------------------------------

    def run_leave_one_out_analysis(
        self, original_data: List[Dict], use_old_method: bool = False
    ) -> List[Dict]:
        """Run leave-one-out analysis.

        Iteratively excludes each dataset and re-runs the meta-analysis
        on the remaining data.  Supports sequential and parallel modes.

        Args:
            original_data: At least 3 dataset dicts.
            use_old_method: Use legacy Fisher method.

        Returns:
            List of result dicts, one per excluded dataset.

        Raises:
            ValueError: If fewer than 3 datasets.
        """
        if len(original_data) <= 2:
            raise ValueError("Need at least 3 datasets for leave-one-out analysis")

        n_datasets = len(original_data)
        max_cores = min(self.max_cores, n_datasets)

        logger.info("Running leave-one-out analysis with %d cores...", max_cores)

        # Build each leave-one-out subset up front. Previously this was
        # computed lazily inside a nested `analyze_subset(i)` closure, which
        # is exactly the kind of object `ProcessPoolExecutor` cannot pickle
        # (see `_run_loo_subset` above): the fix is to hand the executor the
        # module-level `_run_loo_subset` function instead, for both the
        # sequential and parallel branches, so their behavior stays identical.
        subsets = [original_data[:i] + original_data[i + 1 :] for i in range(n_datasets)]

        if max_cores == 1:
            results = []
            for i, subset_data in enumerate(subsets):
                logger.info("Processing subset %d/%d", i + 1, n_datasets)
                results.append(_run_loo_subset(self, subset_data, use_old_method))
        else:
            with ProcessPoolExecutor(max_workers=max_cores) as executor:
                future_to_index = {
                    executor.submit(_run_loo_subset, self, subset_data, use_old_method): i
                    for i, subset_data in enumerate(subsets)
                }
                results = [None] * n_datasets
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        results[index] = future.result()
                        logger.info("Completed subset %d/%d", index + 1, n_datasets)
                    except Exception as exc:
                        logger.error("Subset %d generated exception: %s", index + 1, exc)
                        results[index] = None

        return results

    # ---- high-level wrapper --------------------------------------

    def run_meta_analysis(
        self,
        meta_object: Dict,
        run_leave_one_out: bool = True,
        use_old_method: bool = False,
        max_cores: Optional[int] = None,
    ) -> Dict:
        """Validate, run meta-analysis, optionally run LOO.

        This is the primary entry point.  It validates the input,
        runs the core analysis, and optionally appends leave-one-out
        results.

        Args:
            meta_object: Dict with ``'originalData'`` key.
            run_leave_one_out: Whether to run LOO analysis.
            use_old_method: Use legacy Fisher method.
            max_cores: Override core limit for this run.

        Returns:
            The *meta_object* dict, augmented with
            ``'metaAnalysis'`` and ``'leaveOneOutAnalysis'`` keys.
        """
        if max_cores is not None:
            self.max_cores = min(max_cores, 10, mp.cpu_count())

        meta_object = self._validator.convert_original_data_names(meta_object)

        is_valid, errors = self._validator.validate_meta_object(meta_object, "Meta", "Pre-Analysis")
        if not is_valid:
            raise ValueError("Error in input object:\n" + "\n".join(errors))

        original_data = list(meta_object["originalData"].values())
        class_checks = [self._validator.check_class_vector(ds) for ds in original_data]
        if not all(class_checks):
            failed = [
                name for name, ok in zip(meta_object["originalData"].keys(), class_checks) if not ok
            ]
            raise ValueError(
                f"Insufficient cases/controls in datasets: "
                f"{', '.join(failed)}. Need at least 2 of each."
            )

        logger.info("Running main meta-analysis...")
        meta_object["metaAnalysis"] = self.run_meta_analysis_core(original_data, use_old_method)

        meta_object["leaveOneOutAnalysis"] = None
        if len(original_data) > 2 and run_leave_one_out:
            logger.info("Running leave-one-out analysis...")
            loo_results = self.run_leave_one_out_analysis(original_data, use_old_method)
            dataset_names = list(meta_object["originalData"].keys())
            meta_object["leaveOneOutAnalysis"] = {
                f"removed_{name}": result
                for name, result in zip(dataset_names, loo_results)
                if result is not None
            }

            # Make a silent-degradation failure mode loud instead of quiet.
            # `filter_genes()` uses only the leave-one-out runs that
            # succeeded, and falls back to the plain filter when none did
            # (see `_usable_loo_tables`). Without this warning, a run where
            # every subset failed (as happened for the pickling bug: see
            # `_run_loo_subset`) would quietly lose its leave-one-out
            # filtering even though `is_leave_one_out=True` was requested.
            n_failed = sum(1 for r in loo_results if r is None)
            if n_failed:
                n_total = len(loo_results)
                logger.warning(
                    "%d/%d leave-one-out subset(s) failed to compute (see the "
                    "'Subset N generated exception' error(s) above for the cause). "
                    "Leave-one-out filtering in filter_genes() will use only the "
                    "%d subset(s) that succeeded, and falls back to the plain "
                    "filter if all subsets failed.",
                    n_failed,
                    n_total,
                    n_total - n_failed,
                )

        return meta_object


# ------------------------------------------------------------------
# MetaAnalysisFactory
# ------------------------------------------------------------------
class MetaAnalysisFactory:
    """Factory for creating pre-configured :class:`MetaAnalyzer` instances.

    Mirrors :class:`DataLoaderFactory` in ``src/preprocessing/loader_factory.py``.

    Supported strategies:

    ================  ================================================
    Strategy          Description
    ================  ================================================
    ``random_effects`` Standard inverse-variance random-effects model
                       (default).
    ``fixed_effects``  Inverse-variance fixed-effects model.
    ================  ================================================

    Examples::

        analyzer = MetaAnalysisFactory.create_analyzer("random_effects")
        result = analyzer.run_meta_analysis(meta_object)
    """

    _ENGINES = {
        "random_effects": RandomEffectsEngine,
        "fixed_effects": FixedEffectsEngine,
    }

    @staticmethod
    def create_analyzer(
        strategy: str = "random_effects",
        max_cores: Optional[int] = None,
        **kwargs,
    ) -> MetaAnalyzer:
        """Create a pre-configured MetaAnalyzer.

        Args:
            strategy: ``'random_effects'`` (default) or
                ``'fixed_effects'``.
            max_cores: Core limit passed to :class:`MetaAnalyzer`.
            **kwargs: Passed to the engine constructor.

        Returns:
            Configured :class:`MetaAnalyzer`.
        """
        strategy_lower = strategy.lower().replace("-", "_")

        engine_cls = MetaAnalysisFactory._ENGINES.get(strategy_lower)
        if engine_cls is None:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Options: {list(MetaAnalysisFactory._ENGINES.keys())}."
            )

        engine = engine_cls(**kwargs)
        return MetaAnalyzer(max_cores=max_cores, engine=engine)

    @staticmethod
    def available_strategies() -> List[str]:
        """Return the list of currently implemented strategy names."""
        return list(MetaAnalysisFactory._ENGINES.keys())


# ------------------------------------------------------------------
# Convenience function for backward compatibility
# ------------------------------------------------------------------
def run_meta_analysis(
    meta_object: Dict,
    run_leave_one_out_analysis: bool = True,
    max_cores: int = None,
    use_old_method: bool = False,
) -> Dict:
    """Run a meta-analysis (convenience wrapper).

    Creates a default :class:`MetaAnalyzer` and delegates to
    :meth:`MetaAnalyzer.run_meta_analysis`.

    Args:
        meta_object: Dict with ``'originalData'`` key.
        run_leave_one_out_analysis: Whether to run LOO.
        max_cores: Core limit.
        use_old_method: Use legacy Fisher method.

    Returns:
        The augmented *meta_object* dict.
    """
    analyzer = MetaAnalyzer(max_cores=max_cores)
    return analyzer.run_meta_analysis(
        meta_object=meta_object,
        run_leave_one_out=run_leave_one_out_analysis,
        use_old_method=use_old_method,
    )


def filter_genes(
    meta_object: Dict,
    effect_size_thresh: float = 0.0,
    fdr_thresh: float = 0.05,
    number_studies_thresh: int = 1,
    heterogeneity_pval_thresh: float = 0.0,
    is_leave_one_out: bool = True,
) -> Dict:
    """Filter significant genes from meta-analysis results (convenience wrapper).

    Creates a default :class:`MetaAnalyzer` and delegates to
    :meth:`MetaAnalyzer.filter_genes`.  This is the top-level
    equivalent of R's ``filterGenes()``.

    Args:
        meta_object: Dict returned by :func:`run_meta_analysis`.
        effect_size_thresh: Minimum absolute effect size.
        fdr_thresh: Maximum FDR threshold.
        number_studies_thresh: Minimum contributing studies.
        heterogeneity_pval_thresh: Minimum heterogeneity p-value.
        is_leave_one_out: Use R's leave-one-out rule when results exist.

    Returns:
        The *meta_object* with ``'filterResults'`` added.
    """
    analyzer = MetaAnalyzer()
    return analyzer.filter_genes(
        meta_object=meta_object,
        effect_size_thresh=effect_size_thresh,
        fdr_thresh=fdr_thresh,
        number_studies_thresh=number_studies_thresh,
        heterogeneity_pval_thresh=heterogeneity_pval_thresh,
        is_leave_one_out=is_leave_one_out,
    )
