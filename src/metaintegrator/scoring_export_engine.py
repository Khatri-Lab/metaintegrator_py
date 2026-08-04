"""
Comprehensive export capabilities for scoring and meta-analysis results.

Highlights:
- Unified export interface for all data types (results, plots, reports)
- Multiple format support (CSV, Excel, JSON, Parquet, HDF5)
- Automated report generation with customizable templates
- Batch export capabilities for multiple analyses
- Compression and optimization for large datasets
- Export validation and integrity checks
- Metadata embedding and provenance tracking
"""

import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template

# Import from consolidated core module
from .scoring_core import (
    # Data structures
    ExportFormat,
    StatisticalSummary,
    # Utilities and validation
    LoggerMixin,
    # Exceptions
)


# ==================================================================
# EXPORT CONFIGURATION
# ==================================================================
class ExportConfig:
    """Configuration for export operations."""

    def __init__(
        self,
        base_output_dir: Union[str, Path] = ".",
        include_metadata: bool = True,
        include_plots: bool = True,
        include_summary: bool = True,
        compress_outputs: bool = False,
        validate_exports: bool = True,
        timestamp_files: bool = True,
        custom_prefix: Optional[str] = None,
    ):
        """Initialize export configuration.

        Args:
            base_output_dir: Base directory for all exports
            include_metadata: Whether to include metadata files
            include_plots: Whether to generate and include plots
            include_summary: Whether to include summary reports
            compress_outputs: Whether to compress output files
            validate_exports: Whether to validate exported files
            timestamp_files: Whether to add timestamps to filenames
            custom_prefix: Custom prefix for output filenames
        """
        self.base_output_dir = Path(base_output_dir)
        self.include_metadata = include_metadata
        self.include_plots = include_plots
        self.include_summary = include_summary
        self.compress_outputs = compress_outputs
        self.validate_exports = validate_exports
        self.timestamp_files = timestamp_files
        self.custom_prefix = custom_prefix

        # Create output directory if it doesn't exist
        self.base_output_dir.mkdir(parents=True, exist_ok=True)


# ==================================================================
# EXPORT ENGINE
# ==================================================================
class ExportEngine(LoggerMixin):
    """Comprehensive export engine for meta‑analysis results and visualizations."""

    def __init__(self, config: Optional[ExportConfig] = None, **kwargs):
        """Initialize the export engine.

        Args:
            config: Export configuration object
        """
        super().__init__(**kwargs)
        self.config = config or ExportConfig()
        self.export_registry = {}  # Track exported files

    def export_analysis_results(
        self,
        results: pd.DataFrame,
        analysis_name: str,
        formats: List[Union[str, ExportFormat]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Path]:
        """Export analysis results in multiple formats.

        Args:
            results: DataFrame containing analysis results
            analysis_name: Name for the analysis (used in filenames)
            formats: List of export formats to use
            additional_data: Additional data to export alongside results

        Returns:
            Dictionary mapping format names to output file paths
        """
        if formats is None:
            formats = [ExportFormat.CSV, ExportFormat.EXCEL]

        formats = [ExportFormat(f) if isinstance(f, str) else f for f in formats]

        exported_files = {}

        for fmt in formats:
            try:
                output_path = self._export_dataframe(results, analysis_name, fmt)
                exported_files[fmt.value] = output_path
                self.logger.info(f"Exported {analysis_name} results to {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to export {analysis_name} in {fmt.value} format: {e}")

        if additional_data:
            for data_name, data in additional_data.items():
                try:
                    if isinstance(data, pd.DataFrame):
                        for fmt in formats:
                            output_path = self._export_dataframe(
                                data, f"{analysis_name}_{data_name}", fmt
                            )
                            exported_files[f"{data_name}_{fmt.value}"] = output_path
                    else:
                        output_path = self._export_json(data, f"{analysis_name}_{data_name}")
                        exported_files[f"{data_name}_json"] = output_path
                except Exception as e:
                    self.logger.error(f"Failed to export additional data {data_name}: {e}")

        if self.config.include_metadata:
            metadata_path = self._export_metadata(results, analysis_name, exported_files)
            exported_files["metadata"] = metadata_path

        self.export_registry[analysis_name] = {
            "timestamp": datetime.now(),
            "files": exported_files,
            "formats": [f.value for f in formats],
        }

        return exported_files

    def export_statistical_summary(
        self,
        summary: StatisticalSummary,
        analysis_name: str,
        generate_report: bool = True,
    ) -> Dict[str, Path]:
        """Export statistical summary with optional report generation.

        Args:
            summary: Statistical summary object to export.
            analysis_name: Name for the analysis.
            generate_report: Whether to generate a comprehensive report.

        Returns:
            Dictionary of exported file paths.
        """
        exported_files = {}

        # Export summary data as JSON
        summary_data = {
            "basic_stats": summary.basic_stats,
            "distribution_stats": summary.distribution_stats,
            "correlation_stats": summary.correlation_stats,
            "threshold_counts": summary.threshold_counts,
            "metadata": summary.metadata,
        }

        json_path = self._export_json(summary_data, f"{analysis_name}_statistical_summary")
        exported_files["summary_json"] = json_path

        # Export individual components as CSV/Excel
        if summary.basic_stats:
            basic_stats_df = pd.DataFrame(summary.basic_stats).T
            csv_path = self._export_dataframe(
                basic_stats_df, f"{analysis_name}_basic_stats", ExportFormat.CSV
            )
            exported_files["basic_stats_csv"] = csv_path

        if summary.threshold_counts:
            threshold_df = pd.DataFrame(summary.threshold_counts)
            csv_path = self._export_dataframe(
                threshold_df, f"{analysis_name}_thresholds", ExportFormat.CSV
            )
            exported_files["thresholds_csv"] = csv_path

        # Generate comprehensive report
        if generate_report:
            report_path = self._generate_statistical_report(summary, analysis_name)
            exported_files["report"] = report_path

        # Generate plots if requested
        if self.config.include_plots:
            plot_paths = self._generate_statistical_plots(summary, analysis_name)
            exported_files.update(plot_paths)

        return exported_files

    def export_filter_results(
        self,
        filter_results: Any,
        filter_name: str,
        include_genes: bool = True,
    ) -> Dict[str, Path]:
        """Export filter results with gene lists and summaries.

        Args:
            filter_results: Filter results object to export.
            filter_name: Name for the filter analysis.
            include_genes: Whether to export gene lists.

        Returns:
            Dictionary of exported file paths.
        """
        exported_files = {}

        # Export filter summary
        if is_dataclass(filter_results):
            filter_data = asdict(filter_results)
        elif hasattr(filter_results, "__dict__"):
            filter_data = vars(filter_results)
        else:
            filter_data = filter_results

        json_path = self._export_json(filter_data, f"{filter_name}_filter_results")
        exported_files["filter_json"] = json_path

        # Export gene lists if available and requested
        if include_genes and hasattr(filter_results, "pos_gene_names"):
            genes_data = {
                "positive_genes": getattr(filter_results, "pos_gene_names", []),
                "negative_genes": getattr(filter_results, "neg_gene_names", []),
                "total_positive": len(getattr(filter_results, "pos_gene_names", [])),
                "total_negative": len(getattr(filter_results, "neg_gene_names", [])),
            }

            genes_path = self._export_json(genes_data, f"{filter_name}_gene_lists")
            exported_files["genes_json"] = genes_path

            # Also export as CSV for easy viewing
            all_genes = []
            for gene in getattr(filter_results, "pos_gene_names", []):
                all_genes.append({"gene": gene, "direction": "positive"})
            for gene in getattr(filter_results, "neg_gene_names", []):
                all_genes.append({"gene": gene, "direction": "negative"})

            if all_genes:
                genes_df = pd.DataFrame(all_genes)
                csv_path = self._export_dataframe(
                    genes_df, f"{filter_name}_gene_lists", ExportFormat.CSV
                )
                exported_files["genes_csv"] = csv_path

        return exported_files

    def _export_dataframe(self, df: pd.DataFrame, name: str, format_type: ExportFormat) -> Path:
        """
        Export a pandas DataFrame to a file in the specified format.

        Args:
            df: The DataFrame to export.
            name: Base name for the output file.
            format_type: The format to use for export.

        Returns:
            Path to the exported file.

        Raises:
            ValueError: If the specified format is not supported.
            Exception: For other errors during export.
        """
        filename = self._generate_filename(name, format_type.value)
        filepath = self.config.base_output_dir / filename

        try:
            if format_type == ExportFormat.CSV:
                df.to_csv(filepath, index=False)
            elif format_type == ExportFormat.TSV:
                df.to_csv(filepath, sep="\t", index=False)
            elif format_type == ExportFormat.EXCEL:
                with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name="Data", index=False)
                    # Add metadata sheet if requested
                    if self.config.include_metadata:
                        metadata_df = pd.DataFrame(
                            [
                                {
                                    "export_date": datetime.now(),
                                    "rows": len(df),
                                    "columns": len(df.columns),
                                    "column_names": ", ".join(df.columns.tolist()),
                                }
                            ]
                        )
                        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
            elif format_type == ExportFormat.JSON:
                df.to_json(filepath, orient="records", indent=2)
            elif format_type == ExportFormat.PARQUET:
                df.to_parquet(filepath, index=False)
            elif format_type == ExportFormat.PICKLE:
                df.to_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            # Validate export if requested
            if self.config.validate_exports:
                self._validate_export(filepath, df, format_type)

            return filepath

        except Exception as e:
            self.logger.error(f"Failed to export DataFrame as {format_type.value}: {e}")
            raise

    def _export_json(self, data: Any, name: str) -> Path:
        """
        Export data to a JSON file.

        Args:
            data: The data to serialize to JSON.
            name: Base name for the output file.

        Returns:
            Path to the JSON file.

        Raises:
            Exception: If an error occurs during export.
        """
        filename = self._generate_filename(name, "json")
        filepath = self.config.base_output_dir / filename

        try:
            # Convert numpy types for JSON serialization
            json_data = self._convert_for_json(data)

            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2, default=str)

            return filepath

        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            raise

    def _export_metadata(
        self, df: pd.DataFrame, analysis_name: str, exported_files: Dict[str, Path]
    ) -> Path:
        """
        Export metadata about the analysis and the exported files.

        Args:
            df: The DataFrame that was analysed.
            analysis_name: Name of the current analysis.
            exported_files: Mapping of export type to the file path that was created.

        Returns:
            Path to the JSON file that contains the metadata.
        """
        metadata = {
            "analysis_name": analysis_name,
            "export_timestamp": datetime.now().isoformat(),
            "data_shape": [len(df), len(df.columns)],
            "column_names": df.columns.tolist(),
            "column_types": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "exported_files": {k: str(v) for k, v in exported_files.items()},
            "export_config": {
                "base_output_dir": str(self.config.base_output_dir),
                "include_metadata": self.config.include_metadata,
                "include_plots": self.config.include_plots,
                "compress_outputs": self.config.compress_outputs,
                "validate_exports": self.config.validate_exports,
            },
        }

        return self._export_json(metadata, f"{analysis_name}_metadata")

    def _generate_statistical_plots(
        self, summary: StatisticalSummary, analysis_name: str
    ) -> Dict[str, Path]:
        """
        Generate and save statistical plots for the given analysis.

        Args:
            summary: StatisticalSummary object containing basic statistics and correlations.
            analysis_name: Name of the analysis to use in file names.

        Returns:
            Dictionary mapping plot type to the Path of the generated image file.
        """
        plot_paths = {}

        if not summary.basic_stats:
            return plot_paths

        try:
            # Create plots directory
            plots_dir = self.config.base_output_dir / f"{analysis_name}_plots"
            plots_dir.mkdir(exist_ok=True)

            # Basic statistics overview
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Statistical Overview: {analysis_name}", fontsize=16)

            # Extract data for plotting
            stats_df = pd.DataFrame(summary.basic_stats).T

            # Check if we have any data to plot
            if len(stats_df) == 0:
                plt.close(fig)
                return plot_paths

            # Distribution of means
            if "mean" in stats_df.columns and len(stats_df["mean"].dropna()) > 0:
                mean_data = stats_df["mean"].dropna()
                if len(mean_data) > 0:
                    axes[0, 0].hist(mean_data, bins=min(20, len(mean_data)), alpha=0.7)
                    axes[0, 0].set_title("Distribution of Means")
                    axes[0, 0].set_xlabel("Mean Value")
                    axes[0, 0].set_ylabel("Frequency")
                else:
                    axes[0, 0].text(0.5, 0.5, "No data available", ha="center", va="center")
                    axes[0, 0].set_title("Distribution of Means")
            else:
                axes[0, 0].text(0.5, 0.5, "No mean data available", ha="center", va="center")
                axes[0, 0].set_title("Distribution of Means")

            # Distribution of standard deviations
            if "std" in stats_df.columns and len(stats_df["std"].dropna()) > 0:
                std_data = stats_df["std"].dropna()
                if len(std_data) > 0:
                    axes[0, 1].hist(std_data, bins=min(20, len(std_data)), alpha=0.7)
                    axes[0, 1].set_title("Distribution of Standard Deviations")
                    axes[0, 1].set_xlabel("Standard Deviation")
                    axes[0, 1].set_ylabel("Frequency")
                else:
                    axes[0, 1].text(0.5, 0.5, "No data available", ha="center", va="center")
                    axes[0, 1].set_title("Distribution of Standard Deviations")
            else:
                axes[0, 1].text(0.5, 0.5, "No std data available", ha="center", va="center")
                axes[0, 1].set_title("Distribution of Standard Deviations")

            # Missing data pattern
            if "missing_percent" in stats_df.columns:
                missing_data = stats_df["missing_percent"].fillna(0)
                if len(missing_data) > 0:
                    axes[1, 0].bar(range(len(missing_data)), missing_data)
                    axes[1, 0].set_title("Missing Data Percentage by Variable")
                    axes[1, 0].set_xlabel("Variable Index")
                    axes[1, 0].set_ylabel("Missing Percentage")
                else:
                    axes[1, 0].text(0.5, 0.5, "No missing data info", ha="center", va="center")
                    axes[1, 0].set_title("Missing Data Percentage by Variable")
            else:
                axes[1, 0].text(0.5, 0.5, "No missing data info", ha="center", va="center")
                axes[1, 0].set_title("Missing Data Percentage by Variable")

            # Coefficient of variation
            if "cv" in stats_df.columns:
                cv_data = stats_df["cv"].dropna()
                if len(cv_data) > 0:
                    axes[1, 1].hist(cv_data, bins=min(20, len(cv_data)), alpha=0.7)
                    axes[1, 1].set_title("Distribution of Coefficient of Variation")
                    axes[1, 1].set_xlabel("CV")
                    axes[1, 1].set_ylabel("Frequency")
                else:
                    axes[1, 1].text(0.5, 0.5, "No CV data available", ha="center", va="center")
                    axes[1, 1].set_title("Distribution of Coefficient of Variation")
            else:
                axes[1, 1].text(0.5, 0.5, "No CV data available", ha="center", va="center")
                axes[1, 1].set_title("Distribution of Coefficient of Variation")

            plt.tight_layout()

            # Save overview plot
            overview_path = plots_dir / f"{analysis_name}_statistical_overview.png"
            plt.savefig(overview_path, dpi=300, bbox_inches="tight")
            plt.close()

            plot_paths["overview"] = overview_path

            # Correlation heatmap if available
            if summary.correlation_stats and "pearson_matrix" in summary.correlation_stats:
                corr_matrix = pd.DataFrame(summary.correlation_stats["pearson_matrix"])

                # Only create heatmap if we have data
                if not corr_matrix.empty and corr_matrix.shape[0] > 1:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        cmap="coolwarm",
                        center=0,
                        square=True,
                        fmt=".2f",
                    )
                    plt.title(f"Correlation Matrix: {analysis_name}")
                    plt.tight_layout()

                    corr_path = plots_dir / f"{analysis_name}_correlation_heatmap.png"
                    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    plot_paths["correlation"] = corr_path

        except Exception as e:
            self.logger.error(f"Failed to generate statistical plots: {e}")
            # Close any open figures to prevent memory leaks
            plt.close("all")

        return plot_paths

    def _generate_statistical_report(self, summary: StatisticalSummary, analysis_name: str) -> Path:
        """
        Generate a comprehensive statistical report in Markdown format.

        Args:
            summary: StatisticalSummary object containing all computed statistics.
            analysis_name: Name of the analysis to include in the report title and file name.

        Returns:
            Path to the generated Markdown report file.
        """
        report_template = """
# Statistical Analysis Report: {{ analysis_name }}

Generated on: {{ timestamp }}

## Summary

- **Total Variables Analyzed**: {{ total_variables }}
- **Analysis Type**: Comprehensive Statistical Analysis
- **Report Version**: 2.0.0

## Basic Statistics

{% for variable, stats in basic_stats.items() %}
### {{ variable }}
- **Count**: {{ stats.count }}
- **Mean**: {{ "%.4f"|format(stats.mean) }}
- **Standard Deviation**: {{ "%.4f"|format(stats.std) }}
- **Minimum**: {{ "%.4f"|format(stats.min) }}
- **Maximum**: {{ "%.4f"|format(stats.max) }}
- **Median**: {{ "%.4f"|format(stats.median) }}
- **IQR**: {{ "%.4f"|format(stats.iqr) }}
- **Missing Data**: {{ stats.missing_count }} ({{ "%.1f"|format(stats.missing_percent) }}%)

{% endfor %}

{% if distribution_stats %}
## Distribution Analysis

{% for variable, dist_stats in distribution_stats.items() %}
### {{ variable }}
{% if dist_stats.shapiro_p %}
- **Shapiro-Wilk Test**: p = {{ "%.6f"|format(dist_stats.shapiro_p) }} {% if dist_stats.shapiro_p < 0.05 %}(Non-normal){% else %}(Normal){% endif %}
{% endif %}
- **Skewness**: {{ "%.4f"|format(dist_stats.skewness) }}
- **Kurtosis**: {{ "%.4f"|format(dist_stats.kurtosis) }}
{% if dist_stats.best_distribution %}
- **Best Fit Distribution**: {{ dist_stats.best_distribution.distribution }}
{% endif %}
- **Outliers Detected**: {{ dist_stats.outlier_count }} ({{ "%.1f"|format(dist_stats.outlier_percent) }}%)

{% endfor %}
{% endif %}

{% if correlation_stats %}
## Correlation Analysis

- **Mean Absolute Correlation**: {{ "%.3f"|format(correlation_stats.correlation_summary.mean_abs_correlation) }}
- **Maximum Correlation**: {{ "%.3f"|format(correlation_stats.correlation_summary.max_abs_correlation) }}
- **Strong Correlations (>0.7)**: {{ correlation_stats.correlation_summary.strong_correlations_count }}

{% if correlation_stats.high_correlations %}
### High Correlations (>0.8)
{% for corr in correlation_stats.high_correlations[:5] %}
- **{{ corr.variable1 }} vs {{ corr.variable2 }}**: r = {{ "%.3f"|format(corr.correlation) }}
{% endfor %}
{% endif %}
{% endif %}

## Threshold Analysis

{% for threshold_name, thresholds in threshold_counts.items() %}
### {{ threshold_name }}
{% for threshold, count in thresholds.items() %}
- **{{ threshold }}**: {{ count }} genes
{% endfor %}

{% endfor %}

---
*Report generated by ExportEngine v2.0.0*
        """

        # Prepare template data
        template_data = {
            "analysis_name": analysis_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_variables": len(summary.basic_stats) if summary.basic_stats else 0,
            "basic_stats": summary.basic_stats or {},
            "distribution_stats": summary.distribution_stats or {},
            "correlation_stats": summary.correlation_stats or {},
            "threshold_counts": summary.threshold_counts or {},
        }

        # Generate report
        template = Template(report_template)
        report_content = template.render(**template_data)

        # Save report
        report_path = self.config.base_output_dir / f"{analysis_name}_statistical_report.md"
        with open(report_path, "w") as f:
            f.write(report_content)

        return report_path

    def _create_archive(self, files: List[Path], archive_name: str) -> Path:
        """
        Create a ZIP archive of the specified exported files.

        Args:
            files: List of file paths to include in the archive.
            archive_name: Base name to use for the resulting ZIP file.

        Returns:
            Path to the created ZIP archive.
        """
        archive_path = self.config.base_output_dir / f"{archive_name}_export.zip"

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if file_path.exists():
                    # Use relative path within archive
                    arcname = file_path.relative_to(self.config.base_output_dir)
                    zipf.write(file_path, arcname)

        self.logger.info(f"Created archive: {archive_path}")
        return archive_path

    def _generate_filename(self, base_name: str, extension: str) -> str:
        """
        Generate a standardized filename with optional timestamp and prefix.

        Args:
            base_name: Base name for the file.
            extension: File extension without the leading dot.

        Returns:
            A sanitized filename string.
        """
        filename_parts = []

        if self.config.custom_prefix:
            filename_parts.append(self.config.custom_prefix)

        filename_parts.append(base_name)

        if self.config.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_parts.append(timestamp)

        filename = "_".join(filename_parts) + f".{extension}"

        # Sanitize filename
        filename = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")

        return filename

    def _validate_export(
        self, filepath: Path, original_data: pd.DataFrame, format_type: ExportFormat
    ) -> bool:
        """
        Validate exported file by reading it back and comparing against the original DataFrame.

        Args:
            filepath: Path to the exported file.
            original_data: The original DataFrame that was exported.
            format_type: Format used for export.

        Returns:
            True if the exported file matches the original data, False otherwise.
        """
        try:
            if format_type == ExportFormat.CSV:
                reloaded = pd.read_csv(filepath)
            elif format_type == ExportFormat.TSV:
                reloaded = pd.read_csv(filepath, sep="\t")
            elif format_type == ExportFormat.EXCEL:
                reloaded = pd.read_excel(filepath, sheet_name="Data")
            elif format_type == ExportFormat.JSON:
                # For JSON, just check if file exists and is valid JSON
                with open(filepath, "r") as f:
                    json.load(f)
                self.logger.debug(f"JSON export validation passed for {filepath}")
                return True
            elif format_type == ExportFormat.JSON:
                reloaded = pd.read_json(filepath, orient="records")
            elif format_type == ExportFormat.PICKLE:
                reloaded = pd.read_pickle(filepath)
            else:
                return True  # Skip validation for unsupported formats

            # Basic checks for DataFrame formats
            if format_type != ExportFormat.JSON:
                if reloaded.shape != original_data.shape:
                    self.logger.warning(f"Shape mismatch in exported file {filepath}")
                    return False

                if not reloaded.columns.equals(original_data.columns):
                    self.logger.warning(f"Column mismatch in exported file {filepath}")
                    return False

            self.logger.debug(f"Export validation passed for {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Export validation failed for {filepath}: {e}")
            return False

    def _convert_for_json(self, data: Any) -> Any:
        """
        Convert data types for JSON serialization.

        Args:
            data: Arbitrary data structure that may contain NumPy or pandas types.

        Returns:
            Data structure containing only JSON‑serialisable types.
        """
        if isinstance(data, dict):
            return {k: self._convert_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_for_json(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()
        elif pd.isna(data):
            return None
        else:
            return data

    def get_export_summary(self) -> Dict[str, Any]:
        """
        Get summary of all exports performed by this engine.

        Returns:
            Dictionary containing a count of analyses, the registry itself, and
            the engine configuration used during export.
        """
        return {
            "total_analyses": len(self.export_registry),
            "analyses": self.export_registry,
            "engine_config": {
                "base_output_dir": str(self.config.base_output_dir),
                "include_metadata": self.config.include_metadata,
                "include_plots": self.config.include_plots,
                "compress_outputs": self.config.compress_outputs,
                "validate_exports": self.config.validate_exports,
            },
        }

    def cleanup_exports(self, older_than_days: int = 30) -> int:
        """
        Clean up old export files.

        Args:
            older_than_days: Remove files older than this many days.

        Returns:
            Number of files removed.
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        removed_count = 0
        analyses_to_remove = []

        for analysis_name, export_info in self.export_registry.items():
            if export_info["timestamp"] < cutoff_date:
                # Remove files
                for file_path in export_info["files"].values():
                    if isinstance(file_path, Path) and file_path.exists():
                        file_path.unlink()
                        removed_count += 1

                analyses_to_remove.append(analysis_name)

        # Remove from registry after iteration
        for analysis_name in analyses_to_remove:
            del self.export_registry[analysis_name]

        self.logger.info(f"Cleaned up {removed_count} old export files")
        return removed_count


# ==================================================================
# SPECIALIZED EXPORTERS
# ==================================================================


class ReportExporter(LoggerMixin):
    """Specialized exporter for generating comprehensive analysis reports."""

    def __init__(self, **kwargs):
        """Create a new ReportExporter instance."""
        super().__init__(**kwargs)

    def generate_comprehensive_report(
        self,
        analyses: Dict[str, Any],
        report_name: str,
        output_dir: Path,
    ) -> Path:
        """
        Generate a comprehensive report combining multiple analyses.

        Args:
            analyses: Mapping of analysis identifiers to their data.
            report_name: Base name for the generated report file.
            output_dir: Directory where the report will be written.

        Returns:
            Path to the created report file.
        """
        # Implementation for multi‑analysis reports
        pass


class PlotExporter(LoggerMixin):
    """Specialized exporter for generating and saving plots."""

    def __init__(self, **kwargs):
        """Create a new PlotExporter instance."""
        super().__init__(**kwargs)

    def export_analysis_plots(
        self,
        data: pd.DataFrame,
        plot_types: List[str],
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Generate and export various plot types.

        Args:
            data: DataFrame containing the data to be plotted.
            plot_types: List of plot type identifiers to generate.
            output_dir: Directory where plot images will be saved.

        Returns:
            Mapping of plot type to the path of the exported image file.
        """
        # Implementation for specialized plot generation
        pass
