"""
Extract the raw data behind a violin plot and save it to CSV.

This function mirrors the logic in violin_plot.py's ``_resolve_groups``
and ``_get_or_compute_scores`` to reconstruct the per-sample DataFrame
that the violin plot visualizes.

Usage (inside the notebook, after meta-scores have been computed)::

    from visualization.extract_violin_data import extract_violin_data

    filter_results = meta_analysis_obj['filterResults'].to_dict()

    for ds_name, ds in meta_analysis_obj['originalData'].items():
        df = extract_violin_data(filter_results, ds, label_column='label')
        if df is not None:
            out_path = PLOT_DATA_DIR / f'violin_{ds_name}.csv'
            df.to_csv(out_path, index=True)
            print(f'  Saved: {out_path}')
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def extract_violin_data(
    filter_object: Dict,
    dataset_object: Dict,
    label_column: str = "label",
) -> Optional[pd.DataFrame]:
    """Build a tidy DataFrame with one row per sample.

    Columns
    -------
    Sample     – GEO sample accession  (also used as the index)
    MetaScore  – the meta‑score for that sample
    Class      – binary class label (0 / 1)
    Label      – human‑readable group label

    Parameters
    ----------
    filter_object : dict
        The filter result dict (``posGeneNames`` / ``negGeneNames``).
    dataset_object : dict
        A single dataset dict that **should** already contain a
        ``metaScore`` (or ``score``) key.  If neither key is present
        the function will attempt to compute scores on the fly via
        ``ScoreCalculator``.
    label_column : str
        Column in ``pheno`` used for group labels.  Falls back to
        binary class labels (0 → "Control", 1 → "Disease") when the
        column is missing or has ≤ 1 unique value.

    Returns
    -------
    pd.DataFrame or None
        ``None`` is returned only when scores cannot be obtained.
    """

    # ── 1. Get scores ───────────────────────────────────────────────
    if "metaScore" in dataset_object and dataset_object["metaScore"] is not None:
        scores = np.array(dataset_object["metaScore"], dtype=float)
    elif "score" in dataset_object and dataset_object["score"] is not None:
        scores = np.array(dataset_object["score"], dtype=float)
    else:
        # Compute on the fly if they haven't been cached yet
        try:
            from metaintegrator.score_calculation import ScoreCalculator

            scores = ScoreCalculator.calculate_score(
                filter_object, dataset_object, suppress_messages=True
            )
        except Exception:
            return None

    # ── 2. Get class vector ─────────────────────────────────────────
    classes = np.array(dataset_object["class"], dtype=float)

    # ── 3. Get sample IDs ───────────────────────────────────────────
    samples = list(dataset_object["expr"].columns)

    # ── 4. Resolve group labels (mirrors violin_plot._resolve_groups)
    pheno = dataset_object.get("pheno", None)
    labels = None

    if pheno is not None and isinstance(pheno, pd.DataFrame):
        if label_column in pheno.columns:
            raw_labels = pheno[label_column].values
            unique = pd.Series(raw_labels).dropna().unique()
            if len(unique) > 1:
                labels = [str(l) for l in raw_labels]

    if labels is None:
        class_map = {0: "Control", 1: "Disease"}
        labels = [class_map.get(int(c), f"Class_{int(c)}") for c in classes]

    # ── 5. Assemble DataFrame ───────────────────────────────────────
    df = pd.DataFrame(
        {
            "Sample": samples,
            "MetaScore": scores,
            "Class": classes.astype(int),
            "Label": labels,
        },
        index=samples,
    )
    return df
