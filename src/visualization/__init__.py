# src/visualization/__init__.py

"""
Visualization module for MetaIntegrator Python port.

Provides comprehensive plotting capabilities for meta-analysis results,
including forest plots, heatmaps, ROC/PRC curves, violin plots, and more.

BREAKING CHANGE (naming-collision fix): ``visualization.plotting`` and
``visualization.enhanced_plotting`` used to define two independent classes
that both happened to be named ``MetaAnalysisPlotter`` — different methods,
different modules, no relationship. This package previously re-exported
only the ``enhanced_plotting`` one under the bare name ``MetaAnalysisPlotter``,
so ``from visualization import MetaAnalysisPlotter`` silently got you the
forest/heatmap/ROC/PRC/violin/regression plotter, not the volcano/
effect-size-distribution/study-overview one.

``enhanced_plotting``'s class is now named ``EnhancedMetaAnalysisPlotter``
(with a deprecated, warning-emitting alias for direct
``from visualization.enhanced_plotting import MetaAnalysisPlotter``
imports — see that module). At package level, ``MetaAnalysisPlotter`` now
unambiguously means ``visualization.plotting.MetaAnalysisPlotter``. Code
that did ``from visualization import MetaAnalysisPlotter`` expecting the
enhanced/forest-plot-capable class must switch to
``from visualization import EnhancedMetaAnalysisPlotter``.
"""

# The "basic" plotter: volcano, forest, heatmap, effect-size distribution,
# study overview, MA summary.
from .plotting import MetaAnalysisPlotter

# The "enhanced" plotter: forest/gg-forest/heatmap/Manhattan/
# predictive-value/ROC/PRC/violin/regression, plus multi-dataset ROC/PRC
# convenience functions. Was also named MetaAnalysisPlotter — see the
# module docstring above.
from .enhanced_plotting import (
    EnhancedMetaAnalysisPlotter,
    forest_plot,
    gg_forest_plot,
    heatmap_plot,
    manhattan_plot,
    predictive_value_plot,
    multiple_roc_plot,
    multiple_prc_plot,
    roc_plot,
    prc_plot,
)

from .violin_plot import violin_plot, multi_violin_plot

__all__ = [
    "MetaAnalysisPlotter",
    "EnhancedMetaAnalysisPlotter",
    "forest_plot",
    "gg_forest_plot",
    "heatmap_plot",
    "manhattan_plot",
    "predictive_value_plot",
    "multiple_roc_plot",
    "multiple_prc_plot",
    "roc_plot",
    "prc_plot",
    "violin_plot",
    "multi_violin_plot",
]
