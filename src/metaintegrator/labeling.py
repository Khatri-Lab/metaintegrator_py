"""
Class label assignment for meta-analysis datasets.

This module provides functions for assigning binary class labels
(case vs control) to dataset objects based on phenotype metadata.

Functions:
    assign_class_labels: Assign binary class labels to a dataset.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

_log = logging.getLogger(__name__)

__all__ = [
    "assign_class_labels",
]


def assign_class_labels(
    dataset_object: Dict,
    column_name: str,
    disease_terms: List[str],
    control_terms: Optional[List[str]] = None,
    dataset_name: Optional[str] = None,
) -> Dict:
    """Assign binary class labels (0 = control, 1 = disease) to a dataset.

    Searches a phenotype column for *disease_terms* and marks matching
    samples as **1** (cases).  All other samples are marked **0**
    (controls) unless *control_terms* is provided, in which case only
    samples matching *control_terms* are marked **0** and non-matching
    samples are excluded.

    This is the Python equivalent of R MetaIntegrator's pattern::

        dataset$class <- ifelse(grepl("SLE", dataset$pheno$disease), 1, 0)

    Args:
        dataset_object: Dataset dict that must contain a ``'pheno'``
            DataFrame.
        column_name: Column in ``pheno`` to inspect for matching terms.
        disease_terms: Strings that identify **disease / case** samples.
            Matching is case-insensitive and uses substring search.
        control_terms: Optional strings that identify **control** samples.
            When ``None``, every non-disease sample is treated as control.
        dataset_name: Optional label for log messages.

    Returns:
        The *dataset_object* dict with a ``'class'`` key added (a numpy
        array of 0s and 1s aligned with the pheno rows).

    Raises:
        ValueError: If *column_name* is missing from pheno, or if no
            disease and no control samples are found.

    Examples::

        ds = assign_class_labels(
            dataset,
            column_name="disease state:ch1",
            disease_terms=["SLE", "Lupus"],
            dataset_name="GSE11909_GPL96",
        )
        print(ds["class"])  # array([0., 1., 1., 0., ...])
    """
    if "pheno" not in dataset_object:
        raise ValueError("Dataset must contain a 'pheno' DataFrame")

    pheno = dataset_object["pheno"]
    if column_name not in pheno.columns:
        raise ValueError(
            f"Column '{column_name}' not found in pheno. "
            f"Available: {list(pheno.columns)}"
        )

    label = dataset_name or "dataset"
    values = pheno[column_name].astype(str).str.strip()

    # Case-insensitive substring matching
    disease_mask = values.str.lower().apply(
        lambda v: any(term.lower() in v for term in disease_terms)
    )

    if control_terms is not None:
        control_mask = values.str.lower().apply(
            lambda v: any(term.lower() in v for term in control_terms)
        )
    else:
        control_mask = ~disease_mask

    n_disease = disease_mask.sum()
    n_control = control_mask.sum()

    if n_disease == 0:
        raise ValueError(
            f"{label}: no samples matched disease_terms {disease_terms} "
            f"in column '{column_name}'"
        )
    if n_control == 0:
        raise ValueError(f"{label}: no control samples found in column '{column_name}'")

    class_vec = np.full(len(pheno), np.nan)
    class_vec[disease_mask.values] = 1.0
    class_vec[control_mask.values] = 0.0

    dataset_object["class"] = class_vec

    _log.info(
        "%s: assigned class labels — %d cases, %d controls "
        "(column='%s', disease_terms=%s)",
        label,
        n_disease,
        n_control,
        column_name,
        disease_terms,
    )

    return dataset_object
