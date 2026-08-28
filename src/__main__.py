# src/__main__.py
"""
Package entry point.

Allows:
    python -m src

Subcommands
-----------
coconut_batch_effect_removal
    Run the COCONUT batch-effect removal pipeline.

meta_analysis
    Run the full MetaIntegrator meta-analysis pipeline from a config file.

Examples
--------
Run COCONUT::

    python -m src coconut_batch_effect_removal \\
        --csv_dir data/data_csv --output_dir my_data_out --debug

    # Data with non-default phenotype column names:
    python -m src coconut_batch_effect_removal \\
        --csv_dir data/data_csv --control-col "Group" --disease-col "Case" \\
        --by-platform --platform-col "Batch"

Note: for `coconut_batch_effect_removal`, --by-platform and --no-par-prior
always take their CLI default (False / True) unless passed explicitly, even
when a config file sets `by_platform` or `par_prior` — argparse's store_true /
store_false flags don't have a way to represent "not set" that config-file
merging (see `src/utils/config.py::merge_config`) can distinguish from "set to
the default". Use the CLI flags directly, or call
`src.coconut.coconut_runner.runner()` from Python, for full control.

Run meta-analysis::

    python -m src meta_analysis --config my_config.yaml
    python -m src meta_analysis --config my_config.yaml --dry-run
    python -m src meta_analysis --config my_config.yaml --output-dir ./results --verbose

Generate a config template::

    python -m src meta_analysis --generate-config yaml
    python -m src meta_analysis --generate-config json

Programmatic usage (no subprocess needed)::

    from src.__main__ import run
    run(["meta_analysis", "--config", "my_config.yaml"])

    # Or skip CLI parsing and call the runner directly:
    from src.metaintegrator.meta_analysis_runner import run_pipeline
    result = run_pipeline("my_config.yaml")
"""

import argparse
from typing import List, Optional

from .coconut.coconut_runner import main as coconut_runner_main
from .metaintegrator.meta_analysis_runner import main as meta_analysis_runner_main
from .utils.config import load_config, merge_config


def run(argv: Optional[List[str]] = None) -> None:
    """Parse CLI arguments and dispatch to the matching subcommand handler.

    Args:
        argv: Command-line arguments to parse. Defaults to ``sys.argv[1:]``
            when not given, which is the normal case for real CLI use; pass
            an explicit list for programmatic or test invocations.
    """
    parser = argparse.ArgumentParser(
        prog="Metaintegrator_py",
        description="MetaIntegrator Python — meta-analysis and batch-effect tools",
    )

    parser.add_argument(
        "--config-file",
        help="Path to a global config file (YAML).  Subcommand-specific "
        "settings from this file are merged with CLI flags.",
        default="config/default.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # ==================================================================
    # Subcommand: coconut_batch_effect_removal
    # ==================================================================
    prep_parser = subparsers.add_parser(
        "coconut_batch_effect_removal",
        help="Run COCONUT batch-effect removal on CSV data",
    )
    prep_parser.add_argument(
        "--csv_dir",
        type=str,
        default="data/data_csv",
        help="Directory that contains the input CSV files (default: 'data/data_csv')",
    )
    prep_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional directory to write the output files. "
        "If omitted, results are only printed.",
    )
    prep_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print additional debugging information.",
    )
    prep_parser.add_argument(
        "--control-col",
        type=str,
        default=None,
        dest="control_0_col",
        help="Phenotype column indicating control samples (0 = control). "
        "(default: 'Healthy0.Sepsis1')",
    )
    prep_parser.add_argument(
        "--disease-col",
        type=str,
        default=None,
        dest="disease_col",
        help="Phenotype column indicating disease samples. If omitted, all "
        "non-control samples are treated as disease.",
    )
    prep_parser.add_argument(
        "--by-platform",
        action="store_true",
        default=False,
        dest="by_platform",
        help="Normalize per platform instead of across all datasets at once "
        "(requires --platform-col).",
    )
    prep_parser.add_argument(
        "--platform-col",
        type=str,
        default=None,
        dest="platform_col",
        help="Phenotype column containing platform information, used with "
        "--by-platform. (default: 'Platform_id')",
    )
    prep_parser.add_argument(
        "--no-par-prior",
        action="store_false",
        default=True,
        dest="par_prior",
        help="Disable parametric priors in the Bayesian batch-effect adjustment "
        "(default: enabled).",
    )
    prep_parser.add_argument(
        "--it-conv",
        type=float,
        default=None,
        dest="it_conv",
        help="Convergence threshold for the iterative COCONUT solution. (default: 1e-04)",
    )
    prep_parser.set_defaults(func=coconut_runner_main)

    # ==================================================================
    # Subcommand: meta_analysis
    # ==================================================================
    ma_parser = subparsers.add_parser(
        "meta_analysis",
        help="Run the full MetaIntegrator meta-analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Execute the MetaIntegrator meta-analysis pipeline driven by a\n"
            "JSON or YAML configuration file.\n\n"
            "Quick start:\n"
            "  1.  python -m src meta_analysis --generate-config yaml\n"
            "  2.  Edit meta_analysis_config_template.yaml\n"
            "  3.  python -m src meta_analysis --config meta_analysis_config_template.yaml\n"
        ),
    )
    ma_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a JSON or YAML config file describing the analysis.  "
        "Required unless --generate-config is used.",
    )
    ma_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate the config and check connectivity without running " "the full pipeline.",
    )
    ma_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        dest="output_dir",
        help="Override the output directory specified in the config file.",
    )
    ma_parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging for richer progress output.",
    )
    ma_parser.add_argument(
        "--generate-config",
        type=str,
        choices=["yaml", "json"],
        default=None,
        dest="generate_config",
        metavar="FORMAT",
        help="Generate a documented config template (yaml or json) and exit.",
    )
    ma_parser.set_defaults(func=meta_analysis_runner_main)

    # ==================================================================
    # Parse and dispatch
    # ==================================================================
    args = parser.parse_args(argv)

    cfg = load_config(args.config_file)
    merged_args = merge_config(args, cfg)

    # Dispatch to the subcommand handler
    args.func(merged_args)


def main() -> None:
    """Entry point invoked by ``python -m src``."""
    run()


if __name__ == "__main__":
    main()
