# Potential Weaknesses in R MetaIntegrator

This page collects places where R MetaIntegrator may not do what its
authors intended, or may behave in a way that surprises users.

It is a companion to the [R Parity Register](r_parity_register.md).
The register says where this library differs from R.
This page asks a different question: is the R behavior itself sound?

We write it with care.
R MetaIntegrator is a published, peer-reviewed tool.
Its authors made reasonable choices, often under constraints we do not see.
A finding here is an observation with evidence, not a verdict.

---

## Scope and version

Every entry refers to **R MetaIntegrator 2.1.3**, built and run with
R 4.5.2 for the golden files in this repository.
A later version of R MetaIntegrator may behave differently.
Check an entry again before you rely on it.

---

## Why a separate page

The register has a low bar on purpose.
Any difference from R belongs there, even a tiny one.

This page has a higher bar.
An entry here says something about R itself, so it must be backed by
evidence that a reader can check.
A hunch does not belong here.
A hunch goes in the "Open questions" section at the end, clearly marked.

---

## How to read an entry

### Category

| Category | Meaning |
|---|---|
| **Bug** | The code does something other than what its own comment or documentation says. |
| **Weakness** | The code does what it says, but the rule is fragile or can mislead. |
| **Design choice** | A defensible choice that users should know about. Listed so nobody mistakes it for a bug. |

### Evidence flags

Each entry shows three flags.
A flag is either `yes`, `no`, or `n/a`.

| Flag | Question |
|---|---|
| **Source read** | Did we read the R source for this behavior? |
| **Matches R output** | Does this library reproduce R's own numbers for it (golden files)? |
| **Effect measured** | Did we measure how much it matters? Say whether the data were *simulated* or *real*. |

An entry with `Effect measured: no` makes no claim about size.

### Alternatives

An alternative is a suggestion, not a change.
This library keeps R's behavior as the default, so results match R.
Where we offer an alternative, it is opt-in and named in the entry.
Whether an alternative is *better* is a separate claim.
We make that claim only when we have tested it.

---

## Entries at a glance

| Id | Topic | Category | Source read | Matches R output | Effect measured |
|---|---|---|---|---|---|
| [W-01](#w-01-the-log-scale-rule-reads-one-sample) | The log-scale rule reads one sample | Weakness | yes | yes | simulated only |
| [W-02](#w-02-the-shift-for-negative-values-is-computed-and-not-used) | The shift for negative values is computed and not used | Bug | yes | yes | simulated and 2 golden cases |
| [W-03](#w-03-placeholder-fisher-sum-of-logs) | Fisher `sum_of_logs` | Not assessed | no | no | no |

---

## W-01: The log-scale rule reads one sample

**Category:** Weakness.
**Source read:** yes.
**Matches R output:** yes.
**Effect measured:** simulated data only.

**What R does.**
`.GEM_log_check()` in `getGEOData.R` decides whether a dataset is on a
log scale.
It looks at the first sample column that is not constant and not all
missing.
It compares three normal QQ plots of that column and returns a score.
One sample decides for the whole dataset.

**Why it may mislead.**
Raw intensities with a low median and a narrow spread can look normal
enough to pass as log values.
A dataset where the first sample is unusual can be judged by that sample
alone.

**Evidence.**
The size table, the simulation command and the pinning tests are in
[REG-002](r_parity_register.md#reg-002-rs-rule-for-is-this-data-on-a-log-scale).
In short, for simulated raw lognormal data with a typical intensity of
about 1,100 and 500 values per sample, R left the data untransformed in
34 of 40 draws.
Real series were not measured.

**Alternative (opt-in idea).**
Judge several samples, or a robust summary of the whole matrix, and
report when they disagree.
The web app already shows a warning when R's rule and the former median
rule disagree.
We did not test whether a multi-sample rule is more accurate.

**Upstream.**
Not reported.
See the "Reported upstream" field in REG-002.

---

## W-02: The shift for negative values is computed and not used

**Category:** Bug (the code differs from its own comment).
**Source read:** yes.
**Matches R output:** yes.
**Effect measured:** simulated data, plus the two golden cases that have
negative values.

**What R does.**
In `.GEM_log_check()`, R copies the reference column into `inputC`.
If the minimum of `inputC` is below zero, R adds `-min + 1` to it.
The comment above that block says why:

> make all values positive and apply qq_norm to check if data is log2 normalized

The three `qqnorm()` calls that follow do not read `inputC`.
They read `GEM$expr[, ref_p]`, the column as it was.
So the shifted copy is never used.

**What this library does.**
It matches R.
`gem_log_check()` also decides on the column as given.
A test pins that behavior.

**Do not confuse it with the shift that *is* used.**
Later in `getGEOData.R`, when the check says "not on a log scale", R
shifts the whole matrix by its minimum before `log2()`.
That shift runs.
This entry is only about the one inside the check.

**How much it matters.**
It matters only when the reference column has a negative value.
Two numbers describe the effect.

First, the ten golden inputs.
Two of them have a negative value in the reference column.
The score barely moves, and the decision is the same in both.

| Golden case | Score as R computes it | Score if shifted | Same decision |
|---|---|---|---|
| 05, background subtracted | 1.1877 | 1.1718 | yes |
| 06, log ratios | -2.3900 | -2.7274 | yes |

Second, simulated matrices with 500 values and 3 samples, 200 draws per
row, all with a negative minimum:

| Simulated data | Draws where the decision flips |
|---|---|
| raw lognormal(7, 1.2) minus 300 | 5 of 200 |
| raw lognormal(8, 1.5) minus 500 | 2 of 200 |
| raw lognormal(5, 1.5) minus 100 | 5 of 200 |
| log-scale normal(8, 2) minus 6 | 0 of 200 |
| log ratios normal(0, 1) | 0 of 200 |

The flips are rare in these settings.
We did not measure real series with negative values, such as
background-subtracted Illumina data.

Reproduce with:

```sh
python tests/parity/measurements/measure_register_effects.py unused-shift
```

**Which version is better?**
We do not know.
The comment tells us what the authors intended.
It does not tell us the intended rule is more accurate than the one
that runs.
We have not tested that, so we make no claim.

**Alternative (opt-in idea).**
Apply the shift the comment describes.
This would change decisions in a small share of datasets, so it must not
be the default.
Parity stays the default.

**How to confirm it in R.**
Run `.GEM_log_check()` on a matrix with a negative minimum, once as it
is and once after shifting the column by hand.
Compare the two scores.
Or add a `print(inputC)` after the shift and check that no later line
reads it.

**Upstream.**
Not reported.
We would send a short note to the package maintainer, through the
address in the package's `DESCRIPTION` file.
It would quote the comment, point to the three `qqnorm()` calls, and ask
which behavior is intended.
It would be framed as a question.

**Pinned by.**
`tests/parity/test_register_pins.py::TestW02UnusedShift`.

---

## W-03: Placeholder, Fisher sum of logs

**Category:** Not assessed.
**Source read:** no.
**Matches R output:** no.
**Effect measured:** no.

This entry exists so the question is not lost.
We plan to compare the Fisher `sum_of_logs` method with R's golden
output.
Until then, **we make no claim**.
If the comparison finds nothing, this entry will be removed.

---

## Open questions (not findings)

These are hypotheses.
They have no evidence yet, and nobody should cite them as flaws.

**Q-01: the value of 1 for missing or non-positive gene values.**
`getSampleLevelGeneData()` sets values at or below zero, and missing
values, to 1 before the geometric mean is taken.
That could be a hidden choice with a real effect.

However, `calculateScore()` shifts the whole dataset to positive values
first.
So the practical effect may be small.
We have not tested it.
We will move this question to a numbered entry only if a test shows a
clear effect.

---

## How to add an entry

Follow every step.
Skip none.

1. **Read the R source.** Name the file and function.
   Say what the code does, in neutral words.
2. **Check the comment or documentation.**
   Does the code match what it says?
   If not, it may be a bug.
   If yes, it is a weakness or a design choice.
3. **Reproduce R's numbers.** Use a golden file from R, or add one.
   Set `Matches R output: yes` only when this library reproduces R.
4. **Measure the effect.** Write a command in
   `tests/parity/measurements/` that anyone can run again.
   Say whether the data are simulated or real.
   If you did not measure, write `no`.
5. **Do not claim "better".** Offer an alternative only as opt-in, and
   say whether you tested it.
6. **Keep parity the default.** Add the matching entry to the
   [register](r_parity_register.md) if our behavior differs from R.
7. **Pin it.** Add a test in `tests/parity/test_register_pins.py`.
8. **Pin the version.** State the R MetaIntegrator version you checked.
9. **Decide on upstream.** Write how you would report it.
   Do not put a maintainer's email address on this page.
10. **Check the wording.** Neutral, short sentences, no blame.

---

## References

1. R MetaIntegrator 2.1.3, `R/getGEOData.R`, functions `.GEM_log_check()`
   and the code that calls it.
2. R MetaIntegrator 2.1.3, `R/calculateScore.R`.
3. [R Parity Register](r_parity_register.md), entries REG-002 and REG-003.
4. Golden files: `tests/parity/data/log_check/`.
