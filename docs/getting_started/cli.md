# Command-Line Interface

Everything documented so far in Getting Started calls the Python API directly
(`MetaAnalyzer`, `DataLoaderFactory`, `runner()`, and so on). The package also
ships a command-line interface for running the COCONUT batch-normalization
pipeline and the full meta-analysis pipeline without writing a Python script.

!!! tip "CLI vs. Python API"
    The CLI is a thin wrapper: `coconut_batch_effect_removal` calls
    `src/coconut/coconut_runner.py::runner()`, and `meta_analysis` calls
    `src/metaintegrator/meta_analysis_runner.py::run_pipeline()`. Anything the
    CLI can do, you can also do by importing and calling those functions
    directly: useful if you're integrating this package into a larger
    Python pipeline rather than running it standalone. See
    [Programmatic Usage](#programmatic-usage) below.

## Running the CLI

```bash
python -m src <command> [options]
```

Two subcommands are available: `coconut_batch_effect_removal` and
`meta_analysis`. Run `python -m src --help` or `python -m src <command>
--help` at any time to see the full, current flag list.

## `coconut_batch_effect_removal`

Runs the [COCONUT](../coconut/index.md) batch-normalization pipeline against
a directory of CSV files, using the same code path as
[`coconut_runner.runner()`](../coconut/runner_guide.md).

```bash
python -m src coconut_batch_effect_removal \
    --csv_dir data/data_csv --output_dir my_data_out --debug
```

!!! tip "Try it now: no setup required"
    The repo ships real example data at `data/data_csv` (three GEO sepsis
    studies, pre-trimmed to 2,000 genes), and `config/default.yaml` already
    points `coconut_batch_effect_removal` at that same directory. So from
    the repo root, this runs the full pipeline against real data with
    **zero flags**:

    ```bash
    python -m src coconut_batch_effect_removal
    ```

    That reads all three studies from `data/data_csv`, runs COCONUT with the
    default column names (which match this sample data exactly: see the
    note below), and writes the four combined CSVs to `data/my_data_out`
    (2,000 genes × 30 samples, 15 control / 15 disease). Useful as a smoke
    test after installing, or as a template for your own data.

| Flag | Default | Purpose |
|---|---|---|
| `--csv_dir` | `data/data_csv` | Directory containing the input CSV files. |
| `--output_dir` | *(none: print only)* | Where to write the four combined output CSVs. |
| `--debug` | off | Print verbose progress messages. |
| `--control-col` | `Healthy0.Sepsis1` | Phenotype column indicating control samples (0 = control). |
| `--disease-col` | *(none: all non-control samples)* | Phenotype column indicating disease samples. |
| `--by-platform` | off | Normalize per platform instead of across all datasets at once. |
| `--platform-col` | `Platform_id` | Phenotype column with platform info, used with `--by-platform`. |
| `--no-par-prior` | *(parametric priors on)* | Disable parametric priors in the Bayesian adjustment. |
| `--it-conv` | `1e-04` | Convergence threshold for the iterative solution. |

The column-name defaults (`Healthy0.Sepsis1`, `Platform_id`) match the
column names used by this project's own sample data. **If your phenotype
data uses different column names, pass the matching flags**: otherwise the
run will fail, or silently normalize against the wrong column:

```bash
python -m src coconut_batch_effect_removal \
    --csv_dir data/data_csv \
    --control-col "Group" --disease-col "Case" \
    --by-platform --platform-col "Batch"
```

!!! note "Config file limitation for `--by-platform` / `--no-par-prior`"
    `--csv_dir`, `--control-col`, `--disease-col`, `--platform-col`, and
    `--it-conv` can all be set from the global config file instead of the
    command line (see [Config File](#config-file-coconut_batch_effect_removal)
    below). `--by-platform` and `--no-par-prior` cannot: they're plain
    on/off flags, and argparse has no way to represent "not passed" for
    them that config-file merging can tell apart from "passed with its
    default value." Pass them directly on the command line, or call
    `runner()` from Python for full control.

## `meta_analysis`

Runs the full meta-analysis pipeline: data acquisition, labeling,
meta-analysis, filtering, gene-search optimization, and plotting: from a
single JSON or YAML config file.

**Quick start:**

```bash
# 1. Generate a documented config template
python -m src meta_analysis --generate-config yaml

# 2. Edit my_meta_analysis_config_template.yaml: at minimum, fill in
#    datasets.gse_ids and labeling

# 3. Run it
python -m src meta_analysis --config my_meta_analysis_config_template.yaml
```

| Flag | Purpose |
|---|---|
| `--config PATH` | JSON or YAML config file describing the analysis. Required unless `--generate-config` is used. |
| `--generate-config {yaml,json}` | Write a documented config template to the current directory and exit. |
| `--dry-run` | Validate the config and check connectivity without running the full pipeline. |
| `--output-dir DIR` | Override the output directory set in the config file. |
| `--verbose` | Enable debug-level logging for richer progress output. |

A config file's minimum valid shape is just the GEO accessions to combine
and how to label them as case/control: everything else falls back to a
documented default:

```yaml
datasets:
  gse_ids: ["GSE11909"]
labeling:
  GSE11909:
    disease_terms: ["SLE"]
```

Run `python -m src meta_analysis --generate-config yaml` to see every
available field with inline documentation, including gene filtering
thresholds, forward/backward gene-search options, and which plots to
generate. The generated template, and the validation it's checked against,
live in
[`src/metaintegrator/meta_analysis_config.py`](../api/overview.md).

## Config file (global)

Both subcommands also read a global config file (default:
`config/default.yaml`, overridable with `--config-file`). Values from the
file are merged with CLI flags: **CLI flags take precedence** whenever
they're set to something other than their own default (see the note under
`coconut_batch_effect_removal` above for the one exception).

### Config file: `coconut_batch_effect_removal`

```yaml
coconut_batch_effect_removal:
  csv_dir: data/data_csv
  output_dir: data/my_data_out
  debug: false
  control_0_col: Healthy0.Sepsis1
  disease_col:               # blank = all non-control samples
  platform_col: Platform_id
  it_conv: 0.0001
```

## Programmatic usage

The CLI is optional. Everything it does is also available as a plain
function call, which is usually the better fit if you're integrating this
package into your own scripts or notebooks rather than running it
standalone:

```python
# Skip argument parsing and call the CLI's dispatcher directly:
from src.__main__ import run
run(["meta_analysis", "--config", "my_config.yaml"])

# Or call the pipeline functions directly:
from src.metaintegrator.meta_analysis_runner import run_pipeline
result = run_pipeline("my_config.yaml")

from src.coconut.coconut_runner import runner
runner(csv_dir="./my_data", output_dir="./results",
       control_0_col="Group", disease_col="Case")
```

## See also

- [COCONUT Runner Guide](../coconut/runner_guide.md): full Python API reference for `coconut_runner`
- [API Reference](../api/overview.md): the broader package API, including `EnhancedMetaAnalysisPlotter`
- [MetaIntegrator Quick Start](quickstart.md): the Python-API walkthrough this page complements
