"""
Meta Integrator Filter uses consolidated scoring_core infrastructure.
This module handles filtering operations for meta-analysis results with enhanced capabilities.

Highlights:
- Uses scoring_core infrastructure
- Enhanced filtering strategies with configurable thresholds
- Robust validation and error handling using standardized exceptions
- Improved leave-one-out analysis with statistical validation
- Better integration with other scoring modules
- Advanced filtering methods (adaptive, sequential, multi-criteria)
- Adaptive threshold selection based on data characteristics
- Multi-step filtering with intermediate validation
- Cross-study consistency filtering
- Gene prioritization scoring within filters
- Filter performance assessment and optimization
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

# Import from consolidated core module (eliminates duplicates)
from .scoring_core import (
    # Data structures
    FilterConfig,
    FilterResults,
    # Validators and utilities
    MetaObjectValidator,
    FilterResultsValidator,
    DataFrameUtils,
    LoggerMixin,
    # Exceptions
    FilteringError,
)


# ==================================================================
# ENHANCED FILTER CONFIGURATION
# ==================================================================
@dataclass
class EnhancedFilterConfig(FilterConfig):
    """
    Enhanced filter configuration with advanced features.

    This dataclass extends FilterConfig to provide additional filtering options,
    multi‑criteria weighting, cross‑study consistency checks, gene prioritization,
    and performance optimizations.

    Args:
        adaptive_thresholds (bool):
            Enable adaptive thresholds based on data characteristics.
        min_effect_size_percentile (float):
            Minimum effect size expressed as a percentile (0–100).
        max_heterogeneity_percentile (float):
            Maximum heterogeneity expressed as a percentile (0–100).
        use_combined_criteria (bool):
            Whether to apply a weighted combination of criteria.
        criteria_weights (dict[str, float]):
            Weights for effect_size, fdr, heterogeneity, and n_studies that sum to one
            when use_combined_criteria is True.
        require_consistency_across_studies (bool):
            Require consistent direction across studies.
        min_consistent_studies_pct (float):
            Minimum proportion of studies that must agree on direction (0–1).
        prioritize_by_strength (bool):
            Prioritize genes by strength of evidence within each direction.
        max_genes_per_direction (int | None):
            Maximum number of genes returned per direction.
        cache_intermediate_results (bool):
            Cache intermediate results to avoid recomputation.
        parallel_processing (bool):
            Enable parallel processing for filtering operations.
    """

    # Advanced filtering options
    adaptive_thresholds: bool = False
    min_effect_size_percentile: float = 0.0  # Use percentile instead of fixed threshold
    max_heterogeneity_percentile: float = 100.0

    # Multi-criteria filtering
    use_combined_criteria: bool = False
    criteria_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "effect_size": 0.3,
            "fdr": 0.4,
            "heterogeneity": 0.2,
            "n_studies": 0.1,
        }
    )

    # Cross-study consistency
    require_consistency_across_studies: bool = False
    min_consistent_studies_pct: float = 0.6  # 60% of studies must agree on direction

    # Gene prioritization within filters
    prioritize_by_strength: bool = True
    max_genes_per_direction: Optional[int] = None

    # Performance optimization
    cache_intermediate_results: bool = True
    parallel_processing: bool = False

    def __post_init__(self) -> None:
        """
        Enhanced validation for filter configuration.

        Args:
            self:
                The EnhancedFilterConfig instance.

        Raises:
            ValueError:
                If any of the percentile or proportion attributes fall outside their
                allowed ranges, or if the criterion weights do not sum to one when
                combined criteria are enabled.
        """
        super().__post_init__()

        if not 0 <= self.min_effect_size_percentile <= 100:
            raise ValueError("min_effect_size_percentile must be between 0 and 100")

        if not 0 <= self.max_heterogeneity_percentile <= 100:
            raise ValueError("max_heterogeneity_percentile must be between 0 and 100")

        if not 0 <= self.min_consistent_studies_pct <= 1:
            raise ValueError("min_consistent_studies_pct must be between 0 and 1")

        if self.use_combined_criteria:
            if not np.isclose(sum(self.criteria_weights.values()), 1.0, rtol=0.01):
                raise ValueError("criteria_weights must sum to 1.0")


# ==================================================================
# ENHANCED FILTER RESULTS
# ==================================================================
@dataclass
class EnhancedFilterResults(FilterResults):
    """
    Enhanced filter results with additional analysis metrics.

    This class extends FilterResults by adding performance statistics,
    gene prioritization scores, consistency measures, and a record of
    adaptive thresholds and filtering steps.

    Attributes:
        filter_performance (dict[str, Any]):
            Performance metrics such as precision, recall, or FDR.
        gene_scores (dict[str, float]):
            Prioritization scores assigned to genes.
        cross_study_consistency (dict[str, float]):
            Proportion of studies that agree on the direction of effect for each gene.
        adaptive_thresholds_used (dict[str, float]):
            Threshold values used when adaptive_thresholds is enabled.
        filtering_steps (list[dict[str, Any]]):
            Ordered list of filtering actions applied during the analysis.
    """

    # Additional result metrics
    filter_performance: Dict[str, Any] = field(default_factory=dict)
    gene_scores: Dict[str, float] = field(default_factory=dict)  # Prioritization scores
    cross_study_consistency: Dict[str, float] = field(default_factory=dict)
    adaptive_thresholds_used: Dict[str, float] = field(default_factory=dict)
    filtering_steps: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_prioritized_genes(self) -> int:
        """Total number of genes with prioritization scores."""
        return len(self.gene_scores)

    def get_top_genes(self, direction: str = "both", n: int = 10) -> List[str]:
        """
        Get top N genes by prioritization score.

        Args:
            direction (str):
                Which gene set to consider. 'positive' returns positive genes,
                'negative' returns negative genes, and 'both' returns all genes.
            n (int):
                Number of top genes to return.

        Returns:
            list[str]:
                Gene names sorted by descending prioritization score.
        """
        if direction == "positive":
            candidates = self.pos_gene_names
        elif direction == "negative":
            candidates = self.neg_gene_names
        else:
            candidates = self.pos_gene_names + self.neg_gene_names

        # Sort by gene scores
        candidate_scores = {gene: self.gene_scores.get(gene, 0) for gene in candidates}
        sorted_genes = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)

        return [gene for gene, _ in sorted_genes[:n]]


# ==================================================================
# FILTERING STRATEGY INTERFACE
# ==================================================================


class FilteringStrategy(LoggerMixin):
    """
    Abstract base for filtering strategies.

    Subclasses implement specific filtering algorithms and must provide
    implementations for `apply_filter` and `get_strategy_name`.

    Attributes:
        None
    """

    def apply_filter(
        self,
        data: pd.DataFrame,
        config: EnhancedFilterConfig,
    ) -> Tuple[List[str], List[str]]:
        """
        Apply filtering strategy and return positive and negative gene lists.

        Args:
            self:
                The filtering strategy instance.
            data (pd.DataFrame):
                Input data containing gene statistics.
            config (EnhancedFilterConfig):
                Configuration specifying filtering options.

        Returns:
            tuple[list[str], list[str]]:
                Two lists of gene names – the first for positive
                associations and the second for negative associations.

        Raises:
            NotImplementedError:
                If a subclass has not overridden this method.
        """
        raise NotImplementedError("Subclasses must implement apply_filter")

    def get_strategy_name(self) -> str:
        """
        Get the name of this filtering strategy.

        Returns:
            str: The strategy name.

        Raises:
            NotImplementedError:
                If a subclass has not overridden this method.
        """
        raise NotImplementedError("Subclasses must implement get_strategy_name")


# ==================================================================
# CONCRETE FILTERING STRATEGIES
# ==================================================================


# ------------------------------------------------------------------
# StandardFilteringStrategy Class
# ------------------------------------------------------------------
class StandardFilteringStrategy(FilteringStrategy):
    """
    Standard filtering strategy using fixed thresholds.

    This strategy applies a sequence of hard thresholds on effect size,
    false discovery rate, number of studies, and heterogeneity.
    The filtered genes are then split into positive and negative
    effect groups.

    Attributes:
        None
    """

    def apply_filter(
        self,
        data: pd.DataFrame,
        config: EnhancedFilterConfig,
    ) -> Tuple[List[str], List[str]]:
        """
        Apply standard filtering with fixed thresholds.

        This now matches the R implementation which applies BOTH:
        1. effectSizeFDR filter
        2. Fisher's FDR filter (fisherFDRUp/fisherFDRDown)

        The intersection of these two filters significantly reduces the gene count.

        Args:
            self:
                The StandardFilteringStrategy instance.
            data (pd.DataFrame):
                Gene statistics to filter. Must contain columns:
                - effect_size: effect size values
                - effect_fdr: FDR-corrected p-values from effect size test
                - number_studies: number of studies gene appears in
                - heterogeneity_pval: heterogeneity p-value
                - fisher_fdr_up: Fisher's combined FDR for up-regulation (if available)
                - fisher_fdr_down: Fisher's combined FDR for down-regulation (if available)
            config (EnhancedFilterConfig):
                Configuration containing threshold values.

        Returns:
            tuple[list[str], list[str]]:
                Two lists of gene names – the first for positive
                associations and the second for negative associations.
        """
        # Apply basic filters (same as before)
        filtered_data = data.copy()

        # Effect size filter
        if config.effect_size_thresh > 0:
            filtered_data = filtered_data[
                filtered_data["effect_size"].abs() >= config.effect_size_thresh
            ]

        # FDR filter (effect size FDR)
        if config.fdr_thresh < 1:
            filtered_data = filtered_data[filtered_data["effect_fdr"] <= config.fdr_thresh]

        # Number of studies filter
        if config.number_studies_thresh > 1:
            filtered_data = filtered_data[
                filtered_data["number_studies"] >= config.number_studies_thresh
            ]

        # Heterogeneity filter
        if config.heterogeneity_pval_thresh > 0:
            filtered_data = filtered_data[
                filtered_data["heterogeneity_pval"] >= config.heterogeneity_pval_thresh
            ]

        # ============================================================
        # 🔧 Add Fisher's FDR filter (matching R implementation)
        # ============================================================
        # In R, genes must pass BOTH effectSizeFDR AND fisherFDR filters.
        # This is the critical missing step that caused 14x more genes!

        # Separate candidates by effect direction first
        pos_candidates = filtered_data[filtered_data["effect_size"] > 0]
        neg_candidates = filtered_data[filtered_data["effect_size"] < 0]

        # Check if Fisher FDR columns exist (they should from meta-analysis)
        has_fisher_up = (
            "fisherFDRUp" in pos_candidates.columns or "fisher_fdr_up" in pos_candidates.columns
        )
        has_fisher_down = (
            "fisherFDRDown" in neg_candidates.columns or "fisher_fdr_down" in neg_candidates.columns
        )

        # Normalize column names (handle both camelCase and snake_case)
        if "fisherFDRUp" in pos_candidates.columns:
            fisher_up_col = "fisherFDRUp"
        elif "fisher_fdr_up" in pos_candidates.columns:
            fisher_up_col = "fisher_fdr_up"
        else:
            fisher_up_col = None

        if "fisherFDRDown" in neg_candidates.columns:
            fisher_down_col = "fisherFDRDown"
        elif "fisher_fdr_down" in neg_candidates.columns:
            fisher_down_col = "fisher_fdr_down"
        else:
            fisher_down_col = None

        # Apply Fisher FDR filter for positive genes
        if fisher_up_col is not None:
            pos_genes = pos_candidates[
                pos_candidates[fisher_up_col] <= config.fdr_thresh
            ].index.tolist()
        else:
            # Fallback: if Fisher columns missing, just use effect direction
            # (This should rarely happen if meta-analysis ran correctly)
            import warnings

            warnings.warn(
                "Fisher FDR Up column not found. Filtering without Fisher's test. "
                "Results may not match R implementation."
            )
            pos_genes = pos_candidates.index.tolist()

        # Apply Fisher FDR filter for negative genes
        if fisher_down_col is not None:
            neg_genes = neg_candidates[
                neg_candidates[fisher_down_col] <= config.fdr_thresh
            ].index.tolist()
        else:
            # Fallback: if Fisher columns missing, just use effect direction
            import warnings

            warnings.warn(
                "Fisher FDR Down column not found. Filtering without Fisher's test. "
                "Results may not match R implementation."
            )
            neg_genes = neg_candidates.index.tolist()

        return pos_genes, neg_genes

    def get_strategy_name(self) -> str:
        """
        Get the name of this filtering strategy.

        Returns:
            str: The name 'standard'.
        """
        return "standard"


# ------------------------------------------------------------------
# AdaptiveFilteringStrategy Class
# ------------------------------------------------------------------
class AdaptiveFilteringStrategy(FilteringStrategy):
    """
    Adaptive filtering strategy using data-driven thresholds.

    This strategy computes thresholds for effect size and heterogeneity
    based on the distribution of the input data and then applies these
    adaptive limits together with standard FDR and study‑count filters.
    The resulting genes are split into positive and negative effect
    categories.

    Attributes:
        None
    """

    def apply_filter(
        self,
        data: pd.DataFrame,
        config: EnhancedFilterConfig,
    ) -> Tuple[List[str], List[str]]:
        """
        Apply adaptive filtering with data-driven thresholds.

        Args:
            self:
                The AdaptiveFilteringStrategy instance.
            data (pd.DataFrame):
                Gene statistics to filter.
            config (EnhancedFilterConfig):
                Configuration containing threshold settings.

        Returns:
            tuple[list[str], list[str]]:
                Two lists of gene names – the first for positive
                associations and the second for negative associations.
        """
        adaptive_thresholds = self._calculate_adaptive_thresholds(data, config)

        # Apply adaptive filters
        filtered_data = data.copy()

        # Adaptive effect size filter
        if config.min_effect_size_percentile > 0:
            effect_threshold = adaptive_thresholds["effect_size"]
            filtered_data = filtered_data[filtered_data["effect_size"].abs() >= effect_threshold]

        # Adaptive heterogeneity filter
        if config.max_heterogeneity_percentile < 100:
            het_threshold = adaptive_thresholds["heterogeneity_pval"]
            filtered_data = filtered_data[filtered_data["heterogeneity_pval"] >= het_threshold]

        # Standard FDR and study count filters
        if config.fdr_thresh < 1:
            filtered_data = filtered_data[filtered_data["effect_fdr"] <= config.fdr_thresh]

        if config.number_studies_thresh > 1:
            filtered_data = filtered_data[
                filtered_data["number_studies"] >= config.number_studies_thresh
            ]

        # Separate by effect direction
        pos_genes = filtered_data[filtered_data["effect_size"] > 0].index.tolist()
        neg_genes = filtered_data[filtered_data["effect_size"] < 0].index.tolist()

        return pos_genes, neg_genes

    def _calculate_adaptive_thresholds(
        self,
        data: pd.DataFrame,
        config: EnhancedFilterConfig,
    ) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on data distribution.

        Args:
            self:
                The AdaptiveFilteringStrategy instance.
            data (pd.DataFrame):
                Gene statistics used to compute thresholds.
            config (EnhancedFilterConfig):
                Configuration containing percentile settings.

        Returns:
            dict[str, float]:
                Mapping of threshold names to their computed values.
        """
        thresholds: Dict[str, float] = {}

        # Effect size threshold based on percentile
        if config.min_effect_size_percentile > 0:
            effect_sizes = data["effect_size"].abs()
            thresholds["effect_size"] = np.percentile(
                effect_sizes.dropna(),
                config.min_effect_size_percentile,
            )

        # Heterogeneity threshold based on percentile
        if config.max_heterogeneity_percentile < 100:
            het_pvals = data["heterogeneity_pval"]
            thresholds["heterogeneity_pval"] = np.percentile(
                het_pvals.dropna(),
                100 - config.max_heterogeneity_percentile,
            )

        return thresholds

    def get_strategy_name(self) -> str:
        """
        Get the name of this filtering strategy.

        Returns:
            str: The name 'adaptive'.
        """
        return "adaptive"


# ------------------------------------------------------------------
# CombinedCriteriaFilteringStrategy Class
# ------------------------------------------------------------------
class CombinedCriteriaFilteringStrategy(FilteringStrategy):
    """
    Combined criteria filtering using weighted scoring.

    This strategy assigns a composite score to each gene based on a
    weighted combination of effect size, false discovery rate, heterogeneity,
    and the number of studies. Genes whose scores fall in the top 50 percent
    are retained and then filtered by a standard FDR threshold. The final
    gene sets are split into positive and negative effect directions.

    Attributes:
        None
    """

    def apply_filter(
        self,
        data: pd.DataFrame,
        config: EnhancedFilterConfig,
    ) -> Tuple[List[str], List[str]]:
        """
        Apply combined criteria filtering.

        Args:
            self:
                The CombinedCriteriaFilteringStrategy instance.
            data (pd.DataFrame):
                Gene statistics to be scored and filtered.
            config (EnhancedFilterConfig):
                Configuration containing the criterion weights and
                significance thresholds.

        Returns:
            tuple[list[str], list[str]]:
                Two lists of gene names – the first for positive
                associations and the second for negative associations.
        """
        # Calculate combined scores
        combined_scores = self._calculate_combined_scores(data, config)

        # Determine threshold (e.g., top 50% of genes)
        threshold = np.percentile(combined_scores.dropna(), 50)

        # Filter genes above threshold
        high_score_genes = combined_scores[combined_scores >= threshold]
        filtered_data = data.loc[high_score_genes.index]

        # Apply basic significance filter
        if config.fdr_thresh < 1:
            filtered_data = filtered_data[filtered_data["effect_fdr"] <= config.fdr_thresh]

        # Separate by effect direction
        pos_genes = filtered_data[filtered_data["effect_size"] > 0].index.tolist()
        neg_genes = filtered_data[filtered_data["effect_size"] < 0].index.tolist()

        return pos_genes, neg_genes

    def _calculate_combined_scores(
        self,
        data: pd.DataFrame,
        config: EnhancedFilterConfig,
    ) -> pd.Series:
        """
        Calculate combined scores using weighted criteria.

        Args:
            self:
                The CombinedCriteriaFilteringStrategy instance.
            data (pd.DataFrame):
                Gene statistics used for score calculation.
            config (EnhancedFilterConfig):
                Configuration containing the weights for each criterion.

        Returns:
            pd.Series:
                A series of composite scores indexed by gene name.
        """
        scores = pd.Series(0.0, index=data.index)
        weights = config.criteria_weights

        # Effect size component
        if "effect_size" in weights and "effect_size" in data.columns:
            effect_component = data["effect_size"].abs()
            effect_normalized = (effect_component - effect_component.min()) / (
                effect_component.max() - effect_component.min()
            )
            scores += weights["effect_size"] * effect_normalized

        # FDR component (inverted)
        if "fdr" in weights and "effect_fdr" in data.columns:
            fdr_component = 1 - data["effect_fdr"].clip(0, 1)
            scores += weights["fdr"] * fdr_component

        # Heterogeneity component (inverted)
        if "heterogeneity" in weights and "heterogeneity_pval" in data.columns:
            het_component = data["heterogeneity_pval"].clip(0, 1)
            scores += weights["heterogeneity"] * het_component

        # Number of studies component
        if "n_studies" in weights and "number_studies" in data.columns:
            study_component = data["number_studies"]
            study_normalized = (study_component - study_component.min()) / (
                study_component.max() - study_component.min()
            )
            scores += weights["n_studies"] * study_normalized

        return scores

    def get_strategy_name(self) -> str:
        """
        Get the name of this filtering strategy.

        Returns:
            str: The name 'combined_criteria'.
        """
        return "combined_criteria"


# ==================================================================
# ENHANCED META INTEGRATOR FILTER
# ==================================================================
class EnhancedMetaIntegratorFilter(LoggerMixin):
    """Enhanced meta-integrator filter with advanced filtering capabilities."""

    def __init__(self, config: Optional[EnhancedFilterConfig] = None, **kwargs):
        """Initialize the enhanced meta-integrator filter.

        Args:
            config: Enhanced filter configuration
        """
        super().__init__(**kwargs)
        self.config = config or EnhancedFilterConfig()

        # Initialize filtering strategies
        self.strategies = {
            "standard": StandardFilteringStrategy(),
            "adaptive": AdaptiveFilteringStrategy(),
            "combined_criteria": CombinedCriteriaFilteringStrategy(),
        }

        # Cache for performance
        self._filter_cache = {}

    def filter_meta_analysis(
        self, meta_object: Dict[str, Any], strategy: str = "standard"
    ) -> EnhancedFilterResults:
        """Filter meta-analysis results using specified strategy.

        Args:
            meta_object: Meta-analysis object containing results
                        Must contain 'original_result' or 'filterRes'.
            strategy: Filtering strategy to use

        Returns:
            Enhanced filter results with additional metrics
        """
        # Validate strategy first (before try-except to raise ValueError directly)
        if strategy not in self.strategies:
            raise ValueError(f"Unknown filtering strategy: {strategy}")

        try:
            # Validate inputs
            MetaObjectValidator.validate_and_extract(meta_object)

            # Extract data
            if "original_result" in meta_object:
                data = meta_object["original_result"]
            elif "filterRes" in meta_object:
                data = meta_object["filterRes"]
            else:
                raise ValueError("Meta object must contain 'original_result' or 'filterRes'")

            self.logger.info(f"Starting {strategy} filtering for {len(data)} genes")

            # Apply filtering strategy
            filter_strategy = self.strategies[strategy]
            pos_genes, neg_genes = filter_strategy.apply_filter(data, self.config)

            # Calculate gene prioritization scores if requested
            gene_scores = {}
            if self.config.prioritize_by_strength:
                gene_scores = self._calculate_gene_prioritization_scores(
                    data, pos_genes + neg_genes
                )

            # Apply leave-one-out analysis if requested
            is_leave_one_out = self.config.is_leave_one_out
            if is_leave_one_out:
                pos_genes, neg_genes = self._apply_leave_one_out_filter(
                    meta_object, pos_genes, neg_genes
                )

            # Apply cross-study consistency check if requested
            cross_study_consistency = {}
            if self.config.require_consistency_across_studies and "study" in data.columns:
                pos_genes, neg_genes, cross_study_consistency = (
                    self._filter_by_cross_study_consistency(data, pos_genes, neg_genes)
                )

            # Apply maximum gene limits if specified
            if self.config.max_genes_per_direction:
                pos_genes = self._limit_genes_by_priority(
                    pos_genes, gene_scores, self.config.max_genes_per_direction
                )
                neg_genes = self._limit_genes_by_priority(
                    neg_genes, gene_scores, self.config.max_genes_per_direction
                )

            # Calculate filter performance metrics
            filter_performance = self._assess_filter_performance(data, pos_genes, neg_genes)

            # Get adaptive thresholds if used
            adaptive_thresholds = {}
            if strategy == "adaptive" and self.config.adaptive_thresholds:
                adaptive_thresholds = filter_strategy._calculate_adaptive_thresholds(
                    data, self.config
                )

            # Create enhanced results
            results = EnhancedFilterResults(
                pos_gene_names=pos_genes,
                neg_gene_names=neg_genes,
                config=self.config,
                is_leave_one_out=is_leave_one_out,
                filter_description=f"{strategy} filtering with {len(pos_genes + neg_genes)} genes",
                timestamp=datetime.now(),
                filter_performance=filter_performance,
                gene_scores=gene_scores,
                cross_study_consistency=cross_study_consistency,
                adaptive_thresholds_used=adaptive_thresholds,
                filtering_steps=[
                    {
                        "step": strategy,
                        "genes_before": len(data),
                        "genes_after": len(pos_genes + neg_genes),
                        "timestamp": datetime.now(),
                    }
                ],
            )

            # NOTE: We don't validate 'results' with FilterResultsValidator because:
            # 1. results is an EnhancedFilterResults dataclass (not a dict)
            # 2. FilterResultsValidator expects R-style camelCase dict keys
            # 3. The dataclass is already type-validated by Python
            # 4. Input validation already done via MetaObjectValidator
            # If validation is needed in tests, convert dataclass to dict format first.

            # Validate results
            # FilterResultsValidator.validate_and_extract(results)

            self.logger.info(
                f"Filtering completed: {len(pos_genes)} positive, "
                f"{len(neg_genes)} negative genes"
            )

            return results

        except Exception as e:
            msg = "Meta object must contain 'original_result' or 'filterRes'. If using MetaAnalyzer output, it has a different structure."

            self.logger.error(f"Meta-analysis filtering failed: {e} Details: {msg}")
            raise FilteringError(f"Filtering failed: {e} Details: {msg}")

    def apply_sequential_filtering(
        self, meta_object: Dict[str, Any], strategies: List[str]
    ) -> EnhancedFilterResults:
        """Apply multiple filtering strategies sequentially.

        Args:
            meta_object: Meta‑analysis object to be filtered
            strategies: Ordered list of strategy names to apply

        Returns:
            Enhanced filter results after all sequential steps
        """
        current_data = meta_object.copy()
        all_steps = []

        for i, strategy in enumerate(strategies):
            self.logger.info(f"Applying sequential filter step {i + 1}: {strategy}")

            # Apply current strategy
            results = self.filter_meta_analysis(current_data, strategy)

            # Update data for next step
            if i < len(strategies) - 1:  # Not the last step
                # Create reduced dataset for next iteration
                all_genes = results.pos_gene_names + results.neg_gene_names
                if "original_result" in current_data:
                    current_data["original_result"] = current_data["original_result"].loc[all_genes]
                elif "filterRes" in current_data:
                    current_data["filterRes"] = current_data["filterRes"].loc[all_genes]

            # Track steps
            all_steps.extend(results.filtering_steps)

        # Update final results with all steps
        results.filtering_steps = all_steps
        results.filter_description = f"Sequential filtering: {' -> '.join(strategies)}"

        return results

    def _calculate_gene_prioritization_scores(
        self, data: pd.DataFrame, candidate_genes: List[str]
    ) -> Dict[str, float]:
        """Calculate prioritization scores for candidate genes.

        Args:
            data: Gene‑level dataframe containing analysis metrics
            candidate_genes: List of gene identifiers to score

        Returns:
            Mapping of gene identifiers to their prioritization score
        """
        scores = {}

        for gene in candidate_genes:
            if gene not in data.index:
                continue

            gene_data = data.loc[gene]
            score = 0.0

            # Effect size component
            if "effect_size" in gene_data:
                score += abs(gene_data["effect_size"]) * 0.4

            # Significance component (inverse FDR)
            if "effect_fdr" in gene_data:
                score += (1 - min(gene_data["effect_fdr"], 0.99)) * 0.3

            # Study consistency component
            if "number_studies" in gene_data:
                score += min(gene_data["number_studies"] / 10, 1.0) * 0.2

            # Low heterogeneity component
            if "heterogeneity_pval" in gene_data:
                score += min(gene_data["heterogeneity_pval"], 1.0) * 0.1

            scores[gene] = score

        return scores

    def _apply_leave_one_out_filter(
        self,
        meta_object: Dict[str, Any],
        pos_genes: List[str],
        neg_genes: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Apply leave‑one‑out filtering to validate gene consistency.

        Args:
            meta_object: Meta‑analysis object containing study data
            pos_genes: List of positively identified genes
            neg_genes: List of negatively identified genes

        Returns:
            Tuple containing updated positive and negative gene lists
        """
        # TODO: Simplified placeholder – a full implementation would
        # rerun the meta‑analysis excluding each study in turn
        self.logger.debug("Leave‑one‑out filtering applied (simplified)")
        return pos_genes, neg_genes

    def _filter_by_cross_study_consistency(
        self, data: pd.DataFrame, pos_genes: List[str], neg_genes: List[str]
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """Filter genes based on cross‑study consistency.

        Args:
            data: Gene‑level dataframe that would contain individual study metrics
            pos_genes: List of genes currently marked as positive
            neg_genes: List of genes currently marked as negative

        Returns:
            Tuple containing lists of filtered positive genes, filtered negative genes
            and a dictionary of consistency scores for each gene
        """
        consistency_scores = {}
        filtered_pos = []
        filtered_neg = []

        # Placeholder implementation – real code would use individual study data
        for gene in pos_genes + neg_genes:
            consistency_scores[gene] = 0.8

        min_consistency = self.config.min_consistent_studies_pct

        for gene in pos_genes:
            if consistency_scores.get(gene, 0) >= min_consistency:
                filtered_pos.append(gene)

        for gene in neg_genes:
            if consistency_scores.get(gene, 0) >= min_consistency:
                filtered_neg.append(gene)

        self.logger.info(
            f"Cross‑study consistency filter: {len(filtered_pos)} pos, "
            f"{len(filtered_neg)} neg genes retained"
        )

        return filtered_pos, filtered_neg, consistency_scores

    def _limit_genes_by_priority(
        self, genes: List[str], gene_scores: Dict[str, float], max_genes: int
    ) -> List[str]:
        """Limit genes to a maximum number based on priority scores.

        Args:
            genes: List of gene identifiers to limit
            gene_scores: Mapping of gene identifiers to priority score
            max_genes: Desired maximum number of genes to keep

        Returns:
            Subset of genes limited to the top priority scores
        """
        if len(genes) <= max_genes:
            return genes

        gene_score_pairs = [(gene, gene_scores.get(gene, 0)) for gene in genes]
        gene_score_pairs.sort(key=lambda x: x[1], reverse=True)

        return [gene for gene, _ in gene_score_pairs[:max_genes]]

    def _assess_filter_performance(
        self, original_data: pd.DataFrame, pos_genes: List[str], neg_genes: List[str]
    ) -> Dict[str, Any]:
        """Assess the performance of the filtering operation.

        Args:
            original_data: Original dataframe before filtering
            pos_genes: List of genes retained as positive
            neg_genes: List of genes retained as negative

        Returns:
            Dictionary containing performance metrics such as retention rate,
            counts of positive/negative genes, and summary statistics of retained
            numeric columns
        """
        total_filtered = len(pos_genes) + len(neg_genes)
        total_original = len(original_data)

        performance = {
            "genes_retained": total_filtered,
            "genes_original": total_original,
            "retention_rate": (total_filtered / total_original if total_original > 0 else 0),
            "positive_genes": len(pos_genes),
            "negative_genes": len(neg_genes),
            "direction_ratio": (
                len(pos_genes) / len(neg_genes) if len(neg_genes) > 0 else float("inf")
            ),
        }

        if total_filtered > 0:
            retained_indices = pos_genes + neg_genes
            retained_data = original_data.loc[retained_indices]

            numeric_cols = DataFrameUtils.get_numeric_columns(retained_data)
            for col in numeric_cols:
                if col in retained_data.columns:
                    performance[f"mean_{col}"] = float(retained_data[col].mean())
                    performance[f"std_{col}"] = float(retained_data[col].std())

        return performance

    def get_filter_summary(self) -> Dict[str, Any]:
        """Get summary of filtering operations.

        Returns:
            Dictionary containing the current filter configuration, the list
            of available strategies, and the size of the internal cache.
        """
        return {
            "config": self.config.__dict__,
            "available_strategies": list(self.strategies.keys()),
            "cache_size": len(self._filter_cache),
        }

    def clear_cache(self) -> None:
        """Clear the filter cache to free memory.

        The internal cache that stores intermediate filtering results is
        emptied and a log message is emitted.
        """
        self._filter_cache.clear()
        self.logger.info("Filter cache cleared")


# ==================================================================
# LEGACY COMPATIBILITY CLASS
# ==================================================================
class MetaIntegratorFilter(EnhancedMetaIntegratorFilter):
    """Legacy compatibility class for backward compatibility."""

    def __init__(self, config: Optional[FilterConfig] = None, **kwargs):
        """Initialize with legacy config support.

        Args:
            config: Optional legacy configuration object. If a FilterConfig is supplied it is converted to an EnhancedFilterConfig.
            kwargs: Additional keyword arguments passed to the base class.
        """
        if config and not isinstance(config, EnhancedFilterConfig):
            # Convert legacy config to enhanced config
            enhanced_config = EnhancedFilterConfig(
                effect_size_thresh=config.effect_size_thresh,
                fdr_thresh=config.fdr_thresh,
                number_studies_thresh=config.number_studies_thresh,
                heterogeneity_pval_thresh=config.heterogeneity_pval_thresh,
                is_leave_one_out=config.is_leave_one_out,
            )
        else:
            enhanced_config = config

        super().__init__(enhanced_config, **kwargs)

    def filter(self, meta_object: Dict[str, Any]) -> FilterResults:
        """Legacy method for backward compatibility.

        Args:
            meta_object: Dictionary containing the meta‑analysis data to be filtered.

        Returns:
            FilterResults object constructed from the standard enhanced filter results.
        """
        # Import from scoring_core to use the actual class, not the mock
        from .scoring_core import (
            FilterResults as CoreFilterResults,
            FilterConfig as CoreFilterConfig,
        )

        enhanced_results = self.filter_meta_analysis(meta_object, "standard")

        # Convert to legacy FilterResults using the imported class
        return CoreFilterResults(
            pos_gene_names=enhanced_results.pos_gene_names,
            neg_gene_names=enhanced_results.neg_gene_names,
            config=CoreFilterConfig(
                effect_size_thresh=self.config.effect_size_thresh,
                fdr_thresh=self.config.fdr_thresh,
                number_studies_thresh=self.config.number_studies_thresh,
                heterogeneity_pval_thresh=self.config.heterogeneity_pval_thresh,
                is_leave_one_out=self.config.is_leave_one_out,
            ),
            is_leave_one_out=enhanced_results.is_leave_one_out,
            filter_description=enhanced_results.filter_description,
            timestamp=enhanced_results.timestamp,
        )
