"""
Export functionality for MetaIntegrator summary ROC analysis.

This module provides utilities to export:
- Summary ROC statistics to CSV/Excel
- ROC plots in multiple formats (PNG, PDF, SVG)
- Complete analysis reports with tables and plots

Production-grade implementation with error handling and validation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import openpyxl

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    logger.warning("openpyxl not available. Excel export will be disabled.")


# ------------------------------------------------------------------
# SummaryROCExporter Class
# ------------------------------------------------------------------
class SummaryROCExporter:
    """
    Export summary ROC analysis results to various formats.

    Supports:
    - Statistics export (CSV, Excel)
    - Plot export (PNG, PDF, SVG, EPS)
    - Complete reports with multiple sheets/files
    """

    SUPPORTED_PLOT_FORMATS = ["png", "pdf", "svg", "eps", "jpg", "tiff"]
    SUPPORTED_DATA_FORMATS = ["csv", "excel", "json"]

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Initialize exporter.

        Args:
            output_dir: Directory for exports. If None, uses current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track exported files
        self.exported_files: List[str] = []

    def export_summary_statistics(
        self,
        stats_df: pd.DataFrame,
        filename: str = "summary_roc_stats",
        formats: List[str] = ["csv", "excel"],
    ) -> List[str]:
        """
        Export summary ROC statistics to file(s).

        Args:
            stats_df: DataFrame with summary statistics
            filename: Base filename (without extension)
            formats: List of formats to export ('csv', 'excel', 'json')

        Returns:
            List of exported file paths
        """
        exported = []

        for fmt in formats:
            if fmt not in self.SUPPORTED_DATA_FORMATS:
                logger.warning(f"Unsupported format: {fmt}")
                continue

            try:
                if fmt == "csv":
                    path = self._export_csv(stats_df, filename)
                elif fmt == "excel":
                    path = self._export_excel(stats_df, filename)
                elif fmt == "json":
                    path = self._export_json(stats_df, filename)

                exported.append(path)
                self.exported_files.append(path)
                logger.info(f"Exported to {path}")

            except Exception as e:
                logger.error(f"Failed to export as {fmt}: {str(e)}")

        return exported

    def export_plot(
        self,
        fig: plt.Figure,
        filename: str = "summary_roc_plot",
        formats: List[str] = ["png", "pdf"],
        dpi: int = 300,
        **kwargs,
    ) -> List[str]:
        """
        Export matplotlib figure to file(s).

        Args:
            fig: Matplotlib figure to export
            filename: Base filename (without extension)
            formats: List of formats ('png', 'pdf', 'svg', 'eps', 'jpg', 'tiff')
            dpi: Resolution for raster formats
            **kwargs: Additional arguments for savefig

        Returns:
            List of exported file paths
        """
        exported = []

        # Default savefig parameters
        save_params = {
            "dpi": dpi,
            "bbox_inches": "tight",
            "facecolor": "white",
            "edgecolor": "none",
        }
        save_params.update(kwargs)

        for fmt in formats:
            if fmt not in self.SUPPORTED_PLOT_FORMATS:
                logger.warning(f"Unsupported plot format: {fmt}")
                continue

            try:
                filepath = self.output_dir / f"{filename}.{fmt}"
                fig.savefig(filepath, format=fmt, **save_params)

                exported.append(str(filepath))
                self.exported_files.append(str(filepath))
                logger.info(f"Exported plot to {filepath}")

            except Exception as e:
                logger.error(f"Failed to export plot as {fmt}: {str(e)}")

        return exported

    def export_complete_report(
        self,
        summary_stats: pd.DataFrame,
        study_stats: Optional[pd.DataFrame] = None,
        clinical_params: Optional[pd.DataFrame] = None,
        fig: Optional[plt.Figure] = None,
        report_name: str = "summary_roc_report",
        include_plots: bool = True,
    ) -> str:
        """
        Export complete analysis report with all tables and plots.

        Args:
            summary_stats: Summary ROC statistics (AUC, CI)
            study_stats: Individual study statistics
            clinical_params: Clinical parameters table
            fig: ROC plot figure
            report_name: Base name for report files
            include_plots: Whether to include plot exports

        Returns:
            Path to main report file (Excel if available, else CSV)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_base = f"{report_name}_{timestamp}"

        # Export to Excel if available
        if EXCEL_AVAILABLE:
            return self._export_excel_report(
                summary_stats,
                study_stats,
                clinical_params,
                fig,
                report_base,
                include_plots,
            )
        else:
            # Fallback to multiple CSVs
            return self._export_csv_report(
                summary_stats,
                study_stats,
                clinical_params,
                fig,
                report_base,
                include_plots,
            )

    def _export_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to CSV."""
        filepath = self.output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=True)
        return str(filepath)

    def _export_excel(self, df: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to Excel."""
        if not EXCEL_AVAILABLE:
            raise ImportError("openpyxl required for Excel export")

        filepath = self.output_dir / f"{filename}.xlsx"
        df.to_excel(filepath, index=True, engine="openpyxl")
        return str(filepath)

    def _export_json(self, df: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to JSON."""
        filepath = self.output_dir / f"{filename}.json"
        df.to_json(filepath, orient="index", indent=2)
        return str(filepath)

    def _export_excel_report(
        self,
        summary_stats: pd.DataFrame,
        study_stats: Optional[pd.DataFrame],
        clinical_params: Optional[pd.DataFrame],
        fig: Optional[plt.Figure],
        report_base: str,
        include_plots: bool,
    ) -> str:
        """Export complete report to Excel with multiple sheets."""
        filepath = self.output_dir / f"{report_base}.xlsx"

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Sheet 1: Summary statistics
            summary_stats.to_excel(writer, sheet_name="Summary", index=True)

            # Sheet 2: Individual study statistics
            if study_stats is not None:
                study_stats.to_excel(writer, sheet_name="Study Statistics", index=True)

            # Sheet 3: Clinical parameters
            if clinical_params is not None:
                clinical_params.to_excel(writer, sheet_name="Clinical Parameters", index=True)

            # Sheet 4: Metadata
            metadata = pd.DataFrame(
                {
                    "Item": ["Export Date", "Report Type", "Excel Version"],
                    "Value": [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Summary ROC Analysis",
                        "openpyxl",
                    ],
                }
            )
            metadata.to_excel(writer, sheet_name="Metadata", index=False)

        # Export plots separately if requested
        if include_plots and fig is not None:
            plot_file = f"{report_base}_plot"
            self.export_plot(fig, plot_file, formats=["png", "pdf"])

        self.exported_files.append(str(filepath))
        logger.info(f"Exported complete report to {filepath}")
        return str(filepath)

    def _export_csv_report(
        self,
        summary_stats: pd.DataFrame,
        study_stats: Optional[pd.DataFrame],
        clinical_params: Optional[pd.DataFrame],
        fig: Optional[plt.Figure],
        report_base: str,
        include_plots: bool,
    ) -> str:
        """Export complete report to multiple CSV files."""
        # Create report directory
        report_dir = self.output_dir / report_base
        report_dir.mkdir(exist_ok=True)

        # Export each table
        summary_path = report_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_path, index=True)

        if study_stats is not None:
            study_path = report_dir / "study_statistics.csv"
            study_stats.to_csv(study_path, index=True)

        if clinical_params is not None:
            clinical_path = report_dir / "clinical_parameters.csv"
            clinical_params.to_csv(clinical_path, index=True)

        # Export metadata
        metadata_path = report_dir / "metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Summary ROC Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"\nFiles:\n")
            f.write(f"- summary_statistics.csv\n")
            if study_stats is not None:
                f.write(f"- study_statistics.csv\n")
            if clinical_params is not None:
                f.write(f"- clinical_parameters.csv\n")

        # Export plots
        if include_plots and fig is not None:
            plot_file = str(report_dir / "summary_roc_plot")
            self.export_plot(fig, plot_file, formats=["png", "pdf"])

        main_file = str(summary_path)
        self.exported_files.append(main_file)
        logger.info(f"Exported CSV report to {report_dir}")
        return main_file

    def get_exported_files(self) -> List[str]:
        """Get list of all exported files."""
        return self.exported_files.copy()

    def clear_export_history(self) -> None:
        """Clear the list of exported files."""
        self.exported_files = []


# ------------------------------------------------------------------
# Convenience Functions
# ------------------------------------------------------------------
def export_summary_roc_results(
    summary_stats: pd.DataFrame,
    fig: Optional[plt.Figure] = None,
    output_dir: Optional[str] = None,
    formats: Optional[Dict[str, List[str]]] = None,
    report_name: str = "summary_roc",
) -> Dict[str, List[str]]:
    """
    Export summary ROC results with sensible defaults.

    Args:
        summary_stats: DataFrame with AUC, AUCupper, AUClower
        fig: Optional matplotlib figure
        output_dir: Output directory
        formats: Dict with 'data' and 'plot' format lists
        report_name: Base name for exported files

    Returns:
        Dictionary with 'data' and 'plot' file lists
    """
    if formats is None:
        formats = {
            "data": ["csv", "excel"] if EXCEL_AVAILABLE else ["csv"],
            "plot": ["png", "pdf"],
        }

    exporter = SummaryROCExporter(output_dir)

    results = {}

    # Export data
    results["data"] = exporter.export_summary_statistics(
        summary_stats,
        filename=f"{report_name}_statistics",
        formats=formats.get("data", ["csv"]),
    )

    # Export plot
    if fig is not None:
        results["plot"] = exporter.export_plot(
            fig, filename=f"{report_name}_plot", formats=formats.get("plot", ["png"])
        )
    else:
        results["plot"] = []

    return results


def quick_export(
    summary_stats: pd.DataFrame,
    fig: Optional[plt.Figure] = None,
    filename: str = "summary_roc",
) -> str:
    """
    Quick export to CSV and PNG in current directory.

    Args:
        summary_stats: Summary statistics DataFrame
        fig: Optional plot figure
        filename: Base filename

    Returns:
        Path to CSV file
    """
    exporter = SummaryROCExporter()

    # Export CSV
    csv_path = exporter.export_summary_statistics(
        summary_stats, filename=filename, formats=["csv"]
    )[0]

    # Export plot if provided
    if fig is not None:
        exporter.export_plot(fig, filename=filename, formats=["png"])

    return csv_path
