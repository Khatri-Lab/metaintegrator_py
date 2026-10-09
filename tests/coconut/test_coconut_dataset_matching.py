"""Tests for dataset matching in `coconut()` and `combine_coco_output()`.

Background.
`coconut()` returns a control list and a disease list.
A dataset with no samples on one side is missing from that list.
The old `combine_coco_output()` paired the two lists by position.
So one missing dataset shifted every later pair, and samples were lost silently.

Now every block carries the name of its dataset, and the blocks are paired by name.
These tests cover the names, the by-study guard, and the pairing.
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from coconut.coconut import (
    _merge_dataset_order,
    _process_dataset_pair,
    _resolve_dataset_names,
    coconut,
    combine_coco_output,
)

CONTROL_COL = "cls"
N_GENES = 30


def make_dataset(
    tag: str,
    n_controls: int,
    n_disease: int,
    shift: float = 0.0,
    platform: Optional[str] = None,
    seed: int = 0,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one small dataset whose sample names show their class.

    Control samples are called `<tag>_c0`, `<tag>_c1` and so on.
    Disease samples are called `<tag>_d0`, `<tag>_d1` and so on.
    A test can then check every class label from the sample name alone.

    Args:
        tag: Short label of the dataset, used in the sample names.
        n_controls: Number of control samples.
        n_disease: Number of disease samples.
        shift: Mean of the random values, to mimic a batch effect.
        platform: Platform label. Defaults to `P<tag>`.
        seed: Seed of the random generator.
        name: Optional dataset name to store under the `name` key.

    Returns:
        A dataset dict with `genes` and `pheno`, and `name` if given.
    """
    rng = np.random.default_rng(seed)
    samples = [f"{tag}_c{i}" for i in range(n_controls)] + [f"{tag}_d{i}" for i in range(n_disease)]
    genes = pd.DataFrame(
        rng.normal(shift, 1.0, size=(N_GENES, len(samples))),
        index=[f"G{i}" for i in range(N_GENES)],
        columns=samples,
    )
    pheno = pd.DataFrame(
        {CONTROL_COL: [0] * n_controls + [1] * n_disease, "plat": platform or f"P{tag}"},
        index=samples,
    )
    dataset: Dict[str, Any] = {"genes": genes, "pheno": pheno}
    if name is not None:
        dataset["name"] = name
    return dataset


def complete_datasets() -> List[Dict[str, Any]]:
    """Three datasets that all have controls and disease samples."""
    return [
        make_dataset("A", 4, 4, shift=0.0, seed=1),
        make_dataset("B", 4, 5, shift=1.0, seed=2),
        make_dataset("C", 4, 4, shift=2.0, seed=3),
    ]


def run_by_platform_without_controls(names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run `coconut()` by platform, where dataset B has no controls of its own.

    A and B share platform P1, so B borrows the controls of A.
    C is alone on platform P2.
    The control list then holds A and C, and the disease list holds A, B and C.
    This is the case that broke pairing by position.
    """
    datasets = [
        make_dataset("A", 4, 4, shift=0.0, platform="P1", seed=1),
        make_dataset("B", 0, 5, shift=1.0, platform="P1", seed=2),
        make_dataset("C", 4, 4, shift=2.0, platform="P2", seed=3),
    ]
    if names:
        for dataset, name in zip(datasets, names):
            dataset["name"] = name
    return coconut(datasets, control_0_col=CONTROL_COL, by_platform=True, platform_col="plat")


def strip_names(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of a `coconut()` result whose blocks have no `name`."""

    def without_name(block: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in block.items() if key != "name"}

    return {
        "COCONUTList": [without_name(b) for b in result["COCONUTList"]],
        "rawDiseaseList": [without_name(b) for b in result["rawDiseaseList"]],
        "controlList": {
            **result["controlList"],
            "GSEs": [without_name(b) for b in result["controlList"]["GSEs"]],
        },
    }


def label_from_sample_name(sample: str) -> float:
    """Return the class label that a sample name from `make_dataset()` implies."""
    return 0.0 if "_c" in sample else 1.0


def assert_labels_follow_sample_names(combined: Dict[str, Any]) -> None:
    """Check that each sample got the class label its own name implies."""
    samples = list(combined["genes"].columns)
    expected = [label_from_sample_name(s) for s in samples]
    assert list(combined["class_cntl0_dis1"]) == expected
    assert list(combined["pheno"].index) == samples


# ------------------------------------------------------------------
# Dataset names
# ------------------------------------------------------------------
class TestResolveDatasetNames:
    """Tests for `_resolve_dataset_names()`."""

    def test_default_names_follow_input_order(self):
        """Without names, datasets are called dataset_1, dataset_2 and so on."""
        names = _resolve_dataset_names(complete_datasets())
        assert names == ["dataset_1", "dataset_2", "dataset_3"]

    def test_explicit_names_are_returned_in_input_order(self):
        """Given names are used as they are."""
        datasets = [make_dataset("A", 2, 2, name="GSE9"), make_dataset("B", 2, 2, name="GSE1")]
        assert _resolve_dataset_names(datasets) == ["GSE9", "GSE1"]

    def test_partly_named_datasets_raise(self):
        """Names are all or none, because an auto name could clash with a given one."""
        datasets = [make_dataset("A", 2, 2, name="GSE9"), make_dataset("B", 2, 2)]
        with pytest.raises(ValueError, match="every dataset a non-empty string 'name'"):
            _resolve_dataset_names(datasets)

    def test_duplicate_names_raise(self):
        """Two datasets may not share a name."""
        datasets = [make_dataset("A", 2, 2, name="X"), make_dataset("B", 2, 2, name="X")]
        with pytest.raises(ValueError, match="must be unique"):
            _resolve_dataset_names(datasets)

    @pytest.mark.parametrize("bad_name", ["", "   ", 5])
    def test_blank_or_non_string_names_raise(self, bad_name):
        """A name must be a non-empty string."""
        datasets = [make_dataset("A", 2, 2, name="GSE1"), make_dataset("B", 2, 2)]
        datasets[1]["name"] = bad_name
        with pytest.raises(ValueError, match="Missing or invalid name"):
            _resolve_dataset_names(datasets)


class TestCoconutNamesEveryBlock:
    """`coconut()` must copy the dataset name onto every block it returns."""

    def test_default_names_are_on_every_block(self):
        """Auto names are tied to the input position, not to the list position."""
        result = run_by_platform_without_controls()

        control_names = [b["name"] for b in result["controlList"]["GSEs"]]
        disease_names = [b["name"] for b in result["COCONUTList"]]
        raw_names = [b["name"] for b in result["rawDiseaseList"]]

        # B is the second input dataset, so it stays "dataset_2" although it has no controls.
        assert control_names == ["dataset_1", "dataset_3"]
        assert disease_names == ["dataset_1", "dataset_2", "dataset_3"]
        assert raw_names == disease_names

    def test_user_names_are_on_every_block(self):
        """Names given by the caller are carried through."""
        result = run_by_platform_without_controls(names=["GSE1", "GSE2", "GSE3"])

        assert [b["name"] for b in result["controlList"]["GSEs"]] == ["GSE1", "GSE3"]
        assert [b["name"] for b in result["COCONUTList"]] == ["GSE1", "GSE2", "GSE3"]

    def test_input_datasets_are_not_modified(self):
        """`coconut()` must not add the auto names to the caller's dicts."""
        datasets = complete_datasets()
        keys_before = [set(d) for d in datasets]

        coconut(datasets, control_0_col=CONTROL_COL)

        assert [set(d) for d in datasets] == keys_before


# ------------------------------------------------------------------
# By-study guard
# ------------------------------------------------------------------
class TestByStudyGuard:
    """In by-study mode a dataset may not lack controls or disease samples."""

    def test_dataset_without_controls_raises_and_names_it(self):
        """R COCONUT() stops here as well. The message names the dataset."""
        datasets = complete_datasets()
        datasets[1] = make_dataset("B", 0, 5, shift=1.0, seed=2, name="GSE_B")
        datasets[0]["name"] = "GSE_A"
        datasets[2]["name"] = "GSE_C"

        with pytest.raises(ValueError, match="Dataset 'GSE_B' has < 1 control sample"):
            coconut(datasets, control_0_col=CONTROL_COL)

    def test_dataset_without_disease_samples_raises_and_names_it(self):
        """Without this guard, the disease batches would be numbered wrongly."""
        datasets = complete_datasets()
        datasets[2] = make_dataset("C", 4, 0, shift=2.0, seed=3)

        with pytest.raises(ValueError, match="Dataset 'dataset_3' has < 1 disease sample"):
            coconut(datasets, control_0_col=CONTROL_COL)

    def test_equal_count_mismatch_no_longer_passes_silently(self):
        """B has no controls and C has no disease samples.

        Both lists then have two blocks, so the old code saw nothing wrong
        and paired the wrong datasets.
        """
        datasets = [
            make_dataset("A", 4, 4, seed=1),
            make_dataset("B", 0, 5, seed=2),
            make_dataset("C", 4, 0, seed=3),
        ]
        with pytest.raises(ValueError, match="< 1 control"):
            coconut(datasets, control_0_col=CONTROL_COL)

    def test_by_platform_mode_allows_dataset_without_controls(self):
        """By platform, B borrows the controls of its platform. This must keep working."""
        result = run_by_platform_without_controls()
        assert len(result["COCONUTList"]) == 3


# ------------------------------------------------------------------
# Order of the merged datasets
# ------------------------------------------------------------------
class TestMergeDatasetOrder:
    """Tests for `_merge_dataset_order()`."""

    @pytest.mark.parametrize(
        ("controls", "disease", "expected"),
        [
            (["A", "B", "C"], ["A", "B", "C"], ["A", "B", "C"]),
            (["A", "C"], ["A", "B", "C"], ["A", "B", "C"]),
            (["B", "C"], ["A", "B", "C"], ["A", "B", "C"]),
            (["C"], ["A", "B", "C"], ["A", "B", "C"]),
            (["A", "B"], ["A", "B", "C"], ["A", "B", "C"]),
            (["A"], ["B"], ["A", "B"]),
            (["C", "A"], ["A", "C"], ["C", "A"]),
            ([], ["A", "B"], ["A", "B"]),
            (["A", "B"], [], ["A", "B"]),
        ],
    )
    def test_order_follows_input_order(self, controls, disease, expected):
        """Names missing from the control list are placed next to their disease neighbor."""
        assert _merge_dataset_order(controls, disease) == expected


# ------------------------------------------------------------------
# combine_coco_output()
# ------------------------------------------------------------------
class TestCombineCocoOutputPairsByName:
    """Tests for the pairing done by `combine_coco_output()`."""

    def test_complete_datasets_keep_input_order_and_labels(self):
        """For complete datasets nothing changes: controls first, then disease, per dataset."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # No warning is expected here.
            combined = combine_coco_output(result)

        expected_columns = (
            [f"A_c{i}" for i in range(4)]
            + [f"A_d{i}" for i in range(4)]
            + [f"B_c{i}" for i in range(4)]
            + [f"B_d{i}" for i in range(5)]
            + [f"C_c{i}" for i in range(4)]
            + [f"C_d{i}" for i in range(4)]
        )
        assert list(combined["genes"].columns) == expected_columns
        assert_labels_follow_sample_names(combined)

    def test_dataset_without_controls_keeps_all_samples_and_warns(self):
        """The bug: by position, C's controls met B's disease samples and C's disease vanished."""
        result = run_by_platform_without_controls()

        with pytest.warns(UserWarning, match="Disease samples only: 'dataset_2'"):
            combined = combine_coco_output(result)

        # 8 + 5 + 8 samples. The old code returned 17 and mislabeled B's samples.
        assert combined["genes"].shape == (N_GENES, 21)
        assert len(combined["class_cntl0_dis1"]) == 21
        expected_columns = (
            [f"A_c{i}" for i in range(4)]
            + [f"A_d{i}" for i in range(4)]
            + [f"B_d{i}" for i in range(5)]
            + [f"C_c{i}" for i in range(4)]
            + [f"C_d{i}" for i in range(4)]
        )
        assert list(combined["genes"].columns) == expected_columns
        assert_labels_follow_sample_names(combined)

    def test_dataset_without_controls_gets_normalized_values(self):
        """B's disease samples in the combined matrix must be B's own corrected values."""
        result = run_by_platform_without_controls()
        with pytest.warns(UserWarning):
            combined = combine_coco_output(result)

        corrected_b = result["COCONUTList"][1]["genes"]
        pd.testing.assert_frame_equal(combined["genes"][corrected_b.columns], corrected_b)

    def test_strict_raises_when_a_dataset_is_in_only_one_list(self):
        """With strict=True the caller asks for an error instead of a warning."""
        result = run_by_platform_without_controls(names=["GSE1", "GSE2", "GSE3"])

        with pytest.raises(ValueError, match="Disease samples only: 'GSE2'.*strict=True"):
            combine_coco_output(result, strict=True)

    def test_shuffled_disease_list_gives_the_same_result(self):
        """Pairing uses names, so the list order does not matter."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)
        shuffled = {**result, "COCONUTList": list(reversed(result["COCONUTList"]))}

        expected = combine_coco_output(result)
        actual = combine_coco_output(shuffled)

        pd.testing.assert_frame_equal(actual["genes"], expected["genes"])
        pd.testing.assert_frame_equal(actual["pheno"], expected["pheno"])
        np.testing.assert_array_equal(actual["class_cntl0_dis1"], expected["class_cntl0_dis1"])

    def test_shuffled_control_list_keeps_every_sample_with_its_own_label(self):
        """The output follows the control list order, and each sample keeps its own label."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)
        controls = result["controlList"]
        shuffled = {
            **result,
            "controlList": {**controls, "GSEs": list(reversed(controls["GSEs"]))},
        }

        combined = combine_coco_output(shuffled)

        assert combined["genes"].shape[1] == 4 + 4 + 4 + 5 + 4 + 4
        assert list(combined["genes"].columns[:2]) == ["C_c0", "C_c1"]
        assert_labels_follow_sample_names(combined)

    def test_only_common_pheno_columns_are_kept(self):
        """A column that one dataset lacks is dropped, also with a one-sided dataset."""
        datasets = [
            make_dataset("A", 4, 4, platform="P1", seed=1),
            make_dataset("B", 0, 5, platform="P1", seed=2),
            make_dataset("C", 4, 4, platform="P2", seed=3),
        ]
        datasets[1]["pheno"]["only_in_b"] = "x"
        result = coconut(datasets, control_0_col=CONTROL_COL, by_platform=True, platform_col="plat")

        with pytest.warns(UserWarning):
            combined = combine_coco_output(result)

        assert "only_in_b" not in combined["pheno"].columns
        assert CONTROL_COL in combined["pheno"].columns


class TestCombineCocoOutputUnnamedBlocks:
    """Output made by hand or by an older version has no names on its blocks."""

    def test_equal_length_lists_are_paired_by_position(self):
        """Without names, equal lengths are the only thing we can trust."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)

        expected = combine_coco_output(result)
        actual = combine_coco_output(strip_names(result))

        pd.testing.assert_frame_equal(actual["genes"], expected["genes"])
        np.testing.assert_array_equal(actual["class_cntl0_dis1"], expected["class_cntl0_dis1"])

    def test_unequal_length_lists_raise(self):
        """The old code silently dropped samples here. Now it fails loudly."""
        result = strip_names(run_by_platform_without_controls())

        with pytest.raises(ValueError, match="cannot be paired safely by position"):
            combine_coco_output(result)

    def test_partly_named_blocks_raise(self):
        """A mix of named and unnamed blocks cannot be paired."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)
        del result["COCONUTList"][0]["name"]

        with pytest.raises(ValueError, match="some do not"):
            combine_coco_output(result)

    def test_duplicate_block_names_raise(self):
        """A list may hold at most one block per dataset."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)
        result["COCONUTList"][1]["name"] = result["COCONUTList"][0]["name"]

        with pytest.raises(ValueError, match="disease list has more than one block"):
            combine_coco_output(result)

    def test_invalid_block_name_raises(self):
        """A block name must be a non-empty string."""
        result = coconut(complete_datasets(), control_0_col=CONTROL_COL)
        result["controlList"]["GSEs"][0]["name"] = ""

        with pytest.raises(ValueError, match="invalid dataset name"):
            combine_coco_output(result)

    def test_empty_output_raises(self):
        """There is nothing to combine."""
        empty = {"COCONUTList": [], "rawDiseaseList": [], "controlList": {"GSEs": []}}

        with pytest.raises(ValueError, match="no datasets to combine"):
            combine_coco_output(empty)


class TestProcessDatasetPair:
    """Tests for `_process_dataset_pair()` with a missing side."""

    def test_control_only_dataset_is_labeled_zero(self):
        """A dataset with controls only gets label 0.0 for every sample."""
        block = make_dataset("A", 3, 0)
        merged = _process_dataset_pair(block, None)

        assert list(merged["class_labels"]) == [0.0, 0.0, 0.0]
        assert list(merged["genes"].columns) == ["A_c0", "A_c1", "A_c2"]

    def test_disease_only_dataset_is_labeled_one(self):
        """A dataset with disease samples only gets label 1.0 for every sample."""
        block = make_dataset("B", 0, 2)
        merged = _process_dataset_pair(None, block)

        assert list(merged["class_labels"]) == [1.0, 1.0]

    def test_both_sides_missing_raises(self):
        """At least one side is required."""
        with pytest.raises(ValueError, match="At least one"):
            _process_dataset_pair(None, None)
