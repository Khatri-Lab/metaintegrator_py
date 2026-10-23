# R Parity Register

This page lists every place where metaintegrator_py knowingly differs
from R MetaIntegrator, or knowingly copies a weakness of R.

The goal of the library is to give the same answers as R.
When it does not, you should be able to find out where, why, and how
much it matters.
Nothing on this page is a secret bug.
Each entry says what R does, what we do, the size of the effect, and
the test that keeps the behavior from changing by accident.

This page is about *our* differences from R.
Whether R itself is sound is a separate question with a stricter
evidence bar.
See [Potential Weaknesses in R](r_potential_weaknesses.md).

---

## The policy

We follow four steps, in this order.

1. **Inherit.** Match R, even when R is odd.
   A user who compares our result with an R result should see the same numbers.
2. **Document.** Every known difference gets an entry here.
   A difference that is not written down is a bug.
3. **Measure.** Give the size of the effect as a number, with a command
   anyone can run again.
   "Rare" and "small" are not measurements.
4. **Deviate deliberately.** Differ from R only for a stated reason.
   Count the effect in the output, keep a test that pins the behavior,
   and record the decision here.

The web app follows the same policy.
Its decision record is ADR 0019 in the web app repository.
That record also says which exported files each entry touches.

---

## Entries at a glance

| Id | Topic | Kind | Status |
|---|---|---|---|
| [REG-001](#reg-001-an-exact-zero-before-log2) | An exact zero before `log2` | Deliberate deviation | Decided: keep. Gene-level values now agree with R. |
| [REG-002](#reg-002-rs-rule-for-is-this-data-on-a-log-scale) | R's rule for "is this data on a log scale" | Copied R weakness | Decided: keep parity, warn the user. |
| [REG-003](#reg-003-the-probe-average-and-missing-values) | The probe average and missing values | Unplanned deviation | **Resolved in 0.12.0: now matches R.** |

---

## How to read an entry

| Field | Meaning |
|---|---|
| What R does | The behavior, with the R function that does it. |
| What we do | The behavior here, with the function that does it. |
| Why | The reason for the difference, or why we copy R. |
| How big | The measured effect, and the command that measures it. |
| Pinned by | The tests that fail if the behavior changes. |
| Reported upstream | Whether R's maintainers know, and how we would tell them. |
| Next step | What is left to decide or do. |

### Reporting to R's maintainers

The R maintainers do not use GitHub issues.
The package's `DESCRIPTION` file names a maintainer and has no
`BugReports` field.
So for every entry the field reads "no", with the report we would send.

A report would be an email to the maintainer listed in `DESCRIPTION`.
It would have a subject such as `MetaIntegrator: <short title>`, a
reproducible R snippet of ten lines or fewer, the R and package versions,
what R returns and what we expected, and a link to the entry.
Nothing has been sent yet.

---

## REG-001: An exact zero before log2

**Kind:** Deliberate deviation.

**What R does.**
`getGEOData()` takes `log2` of the whole matrix when the data is not on
a log scale.
`log2(0)` is `-Inf` in R [1].
The matrix keeps the `-Inf`.
`getSampleLevelGeneData()` later sets every value at or below zero to 1,
so `-Inf` ends up as 1 in gene-level values.

**What we do.**
`apply_r_log2()` turns an exact zero into a missing value (`NaN`) and
counts how many it changed.
The count is stored as `zero_values_set_missing` and is shown on the
web app's Results page and in `dataset_preprocessing.csv`.

**Why.**
`-Inf` breaks means, variances and effect sizes further down the
pipeline.
A missing value is skipped by pandas and numpy.
The library has always worked this way.

**How big.**
The difference exists only in datasets that are log2-transformed and
contain exact zeros.
It touches one cell per zero value.
We have not yet counted zeros in real GEO series.
The web app records the count for every run, so real numbers will
accumulate.
Since REG-003 was resolved, the difference no longer reaches gene-level
values: a zero makes the gene 1 in R (through `-Inf`) and here (through
a missing value).
It remains visible in a full "as analyzed" matrix, where a zero is an
empty cell here and `-Inf` in R.

**Pinned by.**

* `tests/parity/test_log_check_parity.py`: compares our matrix with R's
  own output cell by cell (to 1e-9). R's `-Inf` cells must be `NaN` here
  and nowhere else. The R results in the repository were made with
  R 4.5.2 and MetaIntegrator 2.1.3.
* `tests/preprocessing/test_log_scale.py::TestApplyRLog2::test_zeros_become_missing_and_are_counted`.
* `tests/parity/test_register_pins.py::TestReg001ZeroBecomesMissing`.
* `tests/parity/test_register_pins.py::TestReg003ProbeMeanKeepsMissing::test_a_zero_that_log2_made_missing_gives_one_like_r_s_minus_inf`: R's `-Inf` and our `NaN` both give 1.

**Reported upstream.** No.
We would not report it.
`log2(0)` is `-Inf` by definition, and R's later step turns it into 1 on
purpose.
The difference is ours, not a defect in R.

**Next step.**
None for gene-level values.
For the "as analyzed" matrix download (Phase B), write the empty cell
and say so in the caption.

---

## REG-002: R's rule for "is this data on a log scale"

**Kind:** Copied R weakness.

**What R does.**
`.GEM_log_check()` looks at one sample only: the first column that is
not constant and not all missing.
It compares three normal QQ plots of that column (values as they are,
`exp()` of them, and `log(abs(x) + 1e-5)`) and scores them [2].
A score of 0 or below means "already on a log scale".
If the check cannot be computed, R warns and leaves the data alone.

**What we do.**
`gem_log_check()` and `apply_r_log2()` are exact ports.
They reproduce every branch, including R's unused shift of negative
values.
The library's former rule ("log2 when the median is above 20") is kept
only as a comparison, so a change in behavior can be explained.

**Why.**
Parity.
A dataset must be treated the same way in Python and in R.
We copy the weakness on purpose, and the web app warns the user when R's
rule and the median rule disagree.

**How big.**
The rule is sensitive to the shape of one sample.
Raw intensities with a low median and a narrow spread can look
"normal enough" to pass as log values.
The table shows how often R's rule wrongly leaves simulated raw
lognormal intensities untransformed.
Each row has 40 simulated matrices.

| Typical intensity | Values per sample | R leaves raw data as is | Former median rule would transform |
|---|---|---|---|
| about 1,100 | 500 | 34 of 40 | 40 of 40 |
| about 3,000 | 500 | 4 of 40 | 40 of 40 |
| about 3,000 | 20,000 | 0 of 40 | 40 of 40 |
| about 22,000 | 20,000 | 0 of 40 | 40 of 40 |

Measured on 2026-10-20 with a fixed random seed:

```sh
python tests/parity/measurements/measure_register_effects.py fragility
```

These are simulated data.
How often real series fall into the risky zone is not measured.
The web app's yellow warning box marks each real dataset where the two
rules disagree, so the cases will be found when they happen.

**Pinned by.**

* `tests/parity/test_log_check_parity.py`: ten fixed inputs, with R's
  decision, branch, score and correlations.
  The decision must match exactly.
* `tests/parity/test_register_pins.py::TestReg002RLog2Rule`: low raw
  intensities pass as log values; the same shape at higher intensity is
  transformed; only the first usable sample decides.

**Reported upstream.** No.
We would report it as a suggestion, not a bug.
The email would say that `.GEM_log_check()` reads one sample, include
the QQ scores for a raw-intensity matrix that R leaves untransformed,
and suggest checking several samples or a robust summary of the matrix.

**Next step.**
Keep parity and the warning.
Offer a "force log2" or "keep as is" override in the web app only if
scientists ask for it.
It would need its own decision record.

---

## REG-003: The probe average and missing values

**Kind:** Unplanned deviation, found on 2026-10-20 while designing the
signature export. **Resolved on 2026-10-22: the library now matches R.**

**What R does.**
In `R/getSampleLevelGeneData.R`, `.extractDataFromGEM()` calls
`.summarizeSplit()`, which averages the probes of a gene with
`colMeans()` and no `na.rm` [3].
One missing (or `-Inf`) probe value makes the gene value missing (or
`-Inf`) for that sample.
`getSampleLevelGeneData()` then runs `.replaceValues()` (at or below
zero becomes 1), `.replaceNaNs()` and `.replaceNAs()` (missing becomes 1).
So a gene with a missing probe value gets **1** in that sample.
The separate replace steps show that this is intended.

**What we did before 0.12.0.**
Both library copies of the function averaged with pandas, which skips
missing values [4]:

* `metaintegrator/coconut_integration.py`, `SampleLevelGeneExtractor._extract_data_from_gem()`
  (used by the web app's signature export) with `groupby("keys").mean()`.
* `metaintegrator/sample_level_gene_data.py`, `SampleLevelGeneDataProcessor._summarize_split()`
  (used by `score_calculation.py`) with `.mean(axis=0)`.

A gene with probes `[missing, 8]` got **8**, where R gives **1**.

**What we do now.**
Both copies keep the missing value in the average, so the gene becomes
missing and then 1, as in R.
A zero that the log2 step turned into a missing value (REG-001) now
gives 1 as well, the same as R's `-Inf`.

**Why.**
Not a decision at first.
The docstring said the code mirrors R, and nobody had compared the two
on missing probe values.
After it was found, the project owner decided to match R, in line with
the policy of this page.

**How big (the old difference).**
A cell changed only when the gene has two or more probes and at least
one of them is missing, or is an exact zero in a log2-transformed
dataset, in that sample.
Simulated raw data, 200 genes with 3 probes each and 100 samples
(20,000 gene cells), with some exact zeros set at random:

| Share of probe values that are zero | Zero values | Cells that changed | Share of cells |
|---|---|---|---|
| 0 | 0 | 0 | 0.00% |
| 0.01% | 9 | 9 | 0.04% |
| 0.1% | 53 | 53 | 0.27% |
| 1% | 655 | 646 | 3.23% |

```sh
python tests/parity/measurements/measure_register_effects.py signature-cells
```

The column "differ from R" in the tool's output is 0 in every row.
That column is the parity check.
It models R from its source and has not been compared with a live R
session; see the R snippet below.

On your own data (raw probe matrix and a probe-to-gene table), the tool
tells you how many gene cells a run made before 0.12.0 would change:

```sh
python tests/parity/measurements/measure_register_effects.py signature-cells \
    --expr raw_matrix.csv --keys probe_to_gene.csv --genes IFI27,CD3E
```

The table is simulated.
**The share on real series is not measured yet.**

**What it touches.**
Every value built by these functions: the signature expression export
in the web app (`signature_gene_expression.csv`), and the library's
score calculation, which goes through `SampleLevelGeneDataProcessor`.
Runs finished before this change keep the values they were computed
with.
The `value_rule` stored with such a run says that a missing probe value
was skipped.
New runs state the R rule.
Re-running a signature optimization on an old run can give slightly
different scores in the affected cells.

**To confirm the R side.**
Run this once in R and write the result here:

```r
library(MetaIntegrator)
expr <- as.matrix(data.frame(s1 = c(NA, 8, 5), s2 = c(6, 8, 5),
                             row.names = c("p1", "p2", "p3")))
getSampleLevelGeneData(list(expr = expr, keys = c("A", "A", "B")), c("A", "B"))
# From the source we expect gene A, sample s1 = 1, and this library agrees.
```

**Pinned by.**
`tests/parity/test_register_pins.py::TestReg003ProbeMeanKeepsMissing`.
It runs against both library copies of the function.
If it fails, a missing probe value is being skipped again.

**Reported upstream.** No.
We would not report it.
R's behavior is intended (see the replace steps).
The difference was ours, and it is fixed.

**Next step.**
Run the R snippet once.
Optionally, run the measurement tool on two or three real series to see
how many cells in old runs were affected.

---

## Adding an entry

1. Copy an entry above and give it the next number.
2. Fill in every field.
   If a number is not measured, say so in the entry. Do not guess.
3. Add a test that pins the behavior and name it in "Pinned by".
4. Add a row to the table at the top.
5. Add a line to the release notes.

---

## References

1. R documentation, `log`: logarithms and exponentials. https://www.stat.ethz.ch/R-manual/R-devel/library/base/html/Log.html
2. R MetaIntegrator source, `R/getGEOData.R`: `getGEOData()` and `.GEM_log_check()`. The package description points to the method paper at http://biorxiv.org/content/early/2016/08/25/071514.
3. R documentation, `colSums` and `colMeans` (the `na.rm` argument defaults to `FALSE`). https://search.r-project.org/R/refmans/base/html/colSums.html
   R MetaIntegrator source, `R/getSampleLevelGeneData.R`.
4. pandas documentation, `DataFrameGroupBy.mean`. https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.mean.html
5. R documentation, `qqnorm` and `ppoints` (used by the log-scale rule). https://stat.ethz.ch/R-manual/R-devel/library/stats/html/qqnorm.html
