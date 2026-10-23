# Automated R Parity Tests

metaintegrator_py aims to give the same answers as R MetaIntegrator.
The parity tests check that automatically on every test run.
They need no R installation and no network access.

---

## How It Works

A **golden set** is one R `metaObject` saved to CSV files.
It holds both R's inputs and R's outputs.

```text
golden set (from R)                       metaintegrator_py
-------------------                       -----------------
originalData: expr, keys, class  ------>  MetaAnalyzer.run_meta_analysis()
                                                    |
metaAnalysis, leaveOneOutAnalysis  <-- compare -->  Python results
filterResults                      <-- compare -->  MetaAnalyzer.filter_genes()
```

Python reads R's inputs, runs its own analysis, and compares the result with R's saved output.

The tests start **after** data loading.
Both tools get the same expression values, gene keys and class labels.
So a failure here always points at the statistics.
It never comes from GEO downloads or probe annotation.

---

## What Is Checked

| Test | What must match R |
|------|-------------------|
| `test_builtin_golden_sets_exist` | Not a comparison. It fails if the built-in golden data is missing, so the suite cannot silently skip everything. |
| `test_gene_universe_matches` | The set of genes in `pooledResults` |
| `test_pooled_results_match` | All 14 `pooledResults` columns, from `effectSize` to `fisherFDRDown` |
| `test_per_study_values_match` | Per-study effect sizes and standard errors |
| `test_leave_one_out_pooled_results_match` | `pooledResults` of every leave-one-out run |
| `test_standard_filter_matches_r_rule` | `filter_genes()` without leave-one-out, against a port of R's `.filterMetaRun()` |
| `test_standard_filter_matches_r_golden` | `filter_genes()` against gene lists R saved |
| `test_leave_one_out_filter_matches_r_rule` | `filter_genes()` with leave-one-out, against a port of R's `filterGenes()` |

Numbers must agree within `rtol = 1e-8` and `atol = 1e-10`.
Today they agree to about `1e-12`.
So the tolerance absorbs floating-point noise but still catches any real change.

When a test fails, the message names the column, the number of bad rows and the worst gene.
For example:

```text
Parity failed (rtol=1e-08, atol=1e-10):
  effectSize: 167 rows differ (0 NA mismatches), max |diff|=0.00412 at 'Gene13'
```

The "rule" tests use small Python ports of R's filter functions.
They run on R's own tables, so they compare Python with R's logic and R's numbers.
That way every golden set gets filter coverage, even one without saved gene lists.

---

## Running the Tests

```bash
# Everything, as usual
pytest

# Only the parity tests (a few seconds)
pytest -m parity

# Everything except parity
pytest -m "not parity"
```

---

## Golden Sets

### Built in: tinyMetaObject

R MetaIntegrator ships a small example called `tinyMetaObject`.
It has three studies and 167 genes.
Its CSV export is in `data/reference/metaintegrator/input/tinyMetaObject/`.
It is always tested.

One caveat applies.
Its saved `filterResults` were made by an R release from 2015.
They do not follow today's R `filterGenes()` rule.
So the tests that compare against saved gene lists skip this set.
The "rule" tests still cover it.

### Adding your own (for example, the SLE tutorial)

Any `metaObject` can become a golden set in three steps.

**Step 1.** In R, after the analysis, save the object:

```r
save(sleMetaAnalysis, file = "sleMetaAnalysis.RData")
```

**Step 2.** Export it to CSV with the converter that ships in this repository:

```bash
Rscript src/utils/robust_rdata_converter.R sleMetaAnalysis.RData <golden-dir>/sle
```

You may gzip the CSV files (`gzip <golden-dir>/sle/*.csv`).
The loader reads `.csv` and `.csv.gz` alike.
This matters for real studies, which can produce a few hundred MB of CSV.
Large sets are best kept outside the git repository.

**Step 3.** Point the tests at the new directory:

```bash
export MIPY_PARITY_GOLDEN_DIRS=<golden-dir>/sle
pytest -m parity
```

Separate several directories with `:`.
Each directory becomes its own test case, named after the directory.

Re-export the golden set only when you upgrade R MetaIntegrator.
Python changes never need a new export.

---

## Known Gaps

No unexplained gaps are known today.

Differences that we know about are listed in the
[R Parity Register](../comparison/r_parity_register.md).
Each entry names the test that pins it.
Some are kept on purpose, and one is waiting for a decision.

Behaviors of R itself that may surprise users are collected, with evidence
flags, in
[Potential Weaknesses in R](../comparison/r_potential_weaknesses.md).

When a new difference is found and cannot be fixed right away, mark its test
with `pytest.mark.xfail(strict=True, reason="...")`.
The test is then expected to fail and does not break the build.
Once the difference is fixed, the test starts to pass.
`strict=True` then makes it fail on purpose, as a reminder to remove the marker.

The leave-one-out filter was the first gap these tests found.
`filter_genes()` used to filter the main result and then only check that the effect sign stayed the same across leave-one-out runs.
It now follows R: a gene must pass the full filter in every leave-one-out run.

---

## What These Tests Do Not Cover

- **Data loading and annotation.** GEO downloads and probe-to-gene mapping happen before the golden inputs. The SLE notebooks still cover them end to end.
- **Scores, ROC and PRC.** An R `metaObject` does not store per-sample scores. These need their own golden files.
- **COCONUT, immunoStates and LINCS.** Not included yet.

---

## Where the Code Lives

| File | Role |
|------|------|
| `src/utils/r_golden.py` | Loads a golden set and compares tables. No dependency on the analysis code. |
| `tests/parity/parity_support.py` | Finds golden sets and runs the Python analysis once per set. |
| `tests/parity/conftest.py` | Pytest fixtures. |
| `tests/parity/test_meta_analysis_parity.py` | The parity tests and the R filter ports. |
| `tests/parity/test_r_golden_loader.py` | Unit tests for the loader and the comparison helper. |
