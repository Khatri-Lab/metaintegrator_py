# src/__main__.py
"""
Package entry point.

Allows:
    python -m src

Example to call the main function in src/__main__.py programmatically with no subprocess required.

from src.__main__ import run

run([
    "coconut_batch_effect_removal",
    "--csv_dir", "data/data_csv",
    "--output_dir", "data_out_csv",
    "--debug",
])

Example to skip config/CLI logic entirely, call the main function directly:

from src.coconut.coconut_runner import main as coconut_main
coconut_main({
    "csv_dir": "data/data_csv",
    "output_dir": "data_out_csv",
    "debug": True,
})

"""

# src/__main__.py
import argparse

from .utils.config import load_config, merge_config

# from .application.myapp import main as app_main
from .coconut.coconut_runner import main as coconut_runner_main

# from .meta_analysis.meta_analysis import main as meta_analysis_main


# src/__main__.py - plotting commands
def setup_plot_parser(subparsers):
    """
    Sets up subparsers for plotting-related commands and their associated arguments.

    This function defines the structure for various plotting commands within a command-line
    interface tool. It allows users to add subcommands for plot-related operations, such as
    "forest" plots or package-specific plotting styles. Each subcommand is configured with
    the necessary arguments to customize its behavior.

    Args:
        subparsers: The parent argparse subparser object to which the plot parsers will be
            added. It is expected to be a subparsers instance obtained from
            argparse.ArgumentParser.

    Returns:
        None
    """
    plot_parser = subparsers.add_parser("plot")
    plot_subparsers = plot_parser.add_subparsers()

    # Forest plot (works with any meta-analysis results)
    forest_parser = plot_subparsers.add_parser("forest")
    forest_parser.add_argument("--results", required=True)
    forest_parser.add_argument("--style", choices=["classic", "bayesian"], default="classic")

    # Package-specific plotting
    manatee_parser = plot_subparsers.add_parser("manatee-beeswarm")
    # etc...

def run(argv=None):
    """
    Parses command-line arguments and delegates execution flow to the appropriate
    functionality specified through subcommands. This script can be used for
    various types of analyses such as running the main application, performing
    batch effect removal, or running meta-analysis.

    Args:
        argv: Optional. A list of strings representing command-line arguments.
              If None, defaults to `sys.argv`.

    Raises:
        SystemExit: If command-line argument parsing fails or if required
                    arguments are missing.

    Subcommands:
        run: Executes the main application logic.
            --config: Path to a required configuration file.
            --verbose: If set, increases the verbosity of application logs.

        coconut_batch_effect_removal: Removes batch effects using the Coconut tool.
            --csv_dir: Directory containing input CSV files. Default is 'data/data_csv'.
            --output_dir: Optional directory to write output files. If omitted, results are
                          printed to the console.
            --debug: If set, enables additional debugging information. Default is False.

        meta_analysis: Performs meta-analysis on datasets.
            --source: Path to the input source file. Required parameter.
            --format: The input file format, must be either 'csv' or 'json'. Default is 'csv'.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(prog="Metaintegrator_py")

    parser.add_argument(
        "--config-file",
        help="Path to config file",
        default="config/default.yaml",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # ---- run command ----
    """
    run_parser = subparsers.add_parser("run", help="Run main application")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.set_defaults(func=app_main)
    """

    # ---- coconut coconut command ----
    prep_parser = subparsers.add_parser(
        "coconut_batch_effect_removal",
        help="Run coconuting with coconut to remove batch effects",
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
        help="Optional directory to write the three output files. "
        "If omitted, results are only printed.",
    )
    prep_parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="If set, the script prints additional debugging information.",
    )
    prep_parser.set_defaults(func=coconut_runner_main)

    # ---- meta analysis command ----
    """
    manalysis_parser = subparsers.add_parser("meta_analysis", help="Run meta analysis")
    manalysis_parser.add_argument("--source", required=True)
    manalysis_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    manalysis_parser.set_defaults(func=meta_analysis_main
    """

    args = parser.parse_args(argv)

    cfg = load_config(args.config_file)
    merged_args = merge_config(args, cfg)

    # Dispatch
    args.func(merged_args)

def main():
    run()

if __name__ == "__main__":
    main()
