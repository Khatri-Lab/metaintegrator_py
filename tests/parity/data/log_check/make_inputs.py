"""Write the input matrices for the log-scale parity test.

Run from the repository root::

    python tests/parity/data/log_check/make_inputs.py

Each case is a small probes x samples CSV that exercises one branch of R
MetaIntegrator's ``getGEOData()`` log2 decision (``.GEM_log_check``). R
then reads the same files (``tests/parity/r_scripts/make_log_check_golden.R``)
and writes its decisions to ``expected/``. The inputs are fixed by seed,
so rerunning this script reproduces them exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_PROBES = 500


def _frame(values: np.ndarray) -> pd.DataFrame:
    """Probes x samples frame with GEO-style labels."""
    return pd.DataFrame(
        values,
        index=[f"{1000 + i}_at" for i in range(values.shape[0])],
        columns=[f"GSM{100 + j}" for j in range(values.shape[1])],
    )


def build_cases() -> dict[str, pd.DataFrame]:
    """Return every case, keyed by file stem."""
    rng = np.random.default_rng(20261019)
    cases: dict[str, np.ndarray] = {}

    cases["01_raw_lognormal"] = rng.lognormal(8, 1.5, (N_PROBES, 3))
    cases["02_log_scale_normal"] = rng.normal(8, 2, (N_PROBES, 3))
    cases["03_low_intensity_raw"] = rng.lognormal(2.5, 1.5, (N_PROBES, 3))
    cases["04_flat_large_values"] = rng.uniform(100, 10000, (N_PROBES, 3))
    cases["05_background_subtracted"] = rng.lognormal(6, 1.5, (N_PROBES, 3)) - 200
    cases["06_log_ratios_negative"] = rng.normal(0, 2, (N_PROBES, 3))

    first_cols = rng.lognormal(8, 1.5, (N_PROBES, 3))
    first_cols[:, 0] = 7.0  # constant: R skips it
    first_cols[::7, 1] = np.nan  # missing values in the reference column
    cases["07_constant_first_column_and_na"] = first_cols

    cases["08_exp_overflows_everywhere"] = rng.lognormal(9, 1, (N_PROBES, 3)) + 1000
    cases["09_ties_rounded"] = np.round(rng.lognormal(5, 1, (N_PROBES, 3)))

    zeros = rng.lognormal(8, 1.5, (N_PROBES, 3))
    zeros[:10, 1] = 0.0
    cases["10_raw_with_zeros"] = zeros

    return {name: _frame(values) for name, values in cases.items()}


def main() -> None:
    """Write every case to ``inputs/``."""
    out = HERE / "inputs"
    out.mkdir(exist_ok=True)
    for name, frame in build_cases().items():
        frame.to_csv(out / f"{name}.csv", float_format="%.10g")
    print(f"Wrote {len(build_cases())} cases to {out}")


if __name__ == "__main__":
    main()
