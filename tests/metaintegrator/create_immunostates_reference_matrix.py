#!/usr/bin/env python3
"""
Generate a realistic synthetic immunoStates reference matrix.

This creates a high-quality synthetic matrix based on the published immunoStates
structure, using real immune marker genes with biologically plausible expression patterns.

The matrix is suitable for:
- Testing the ReferenceDataManager
- Development and validation
- Demonstrating the deconvolution pipeline
- Educational purposes

Note: This is synthetic data. For actual research, use the real immunoStates matrix.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def create_realistic_immunostates_matrix(
    output_path: str = None, random_seed: int = 42
) -> pd.DataFrame:
    """
    Create a realistic synthetic immunoStates reference matrix.

    Based on the actual immunoStates structure:
    - ~317 signature genes
    - 24 cell types
    - Cell-type specific expression patterns

    Args:
        output_path: Where to save CSV (if None, returns DataFrame only)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with genes as rows, cell types as columns
    """
    np.random.seed(random_seed)

    # Define all 24 cell types from actual immunoStates
    cell_types = [
        "CD56bright_natural_killer_cell",
        "CD56dim_natural_killer_cell",
        "CD14_positive_monocyte",
        "CD16_positive_monocyte",
        "naive_B_cell",
        "memory_B_cell",
        "plasma_cell",
        "CD8_positive_alpha_beta_T_cell",
        "CD4_positive_alpha_beta_T_cell",
        "gamma_delta_T_cell",
        "neutrophil",
        "eosinophil",
        "basophil",
        "myeloid_dendritic_cell",
        "plasmacytoid_dendritic_cell",
        "CD4_Tcm",  # Central memory
        "CD4_Tem",  # Effector memory
        "CD8_Tcm",
        "CD8_Tem",
        "Treg",  # Regulatory T cells
        "Th1",
        "Th2",
        "Th17",
        "Tfh",  # Follicular helper T cells
    ]

    # Real immune marker genes organized by cell type
    signature_genes = {
        # NK cells
        "NK": [
            "NCAM1",
            "KLRD1",
            "KLRB1",
            "KLRC1",
            "KLRC2",
            "NCR1",
            "NCR3",
            "GZMB",
            "PRF1",
            "GNLY",
            "NKG7",
            "FCGR3A",
            "FCGR3B",
        ],
        # T cells (general)
        "T_general": ["CD3D", "CD3E", "CD3G", "CD2", "CD7", "CD5"],
        # CD8 T cells
        "CD8_T": ["CD8A", "CD8B", "GZMK", "CCL5", "CST7"],
        # CD4 T cells
        "CD4_T": ["CD4", "IL7R", "MAL", "LTB"],
        # Regulatory T cells
        "Treg": ["FOXP3", "IL2RA", "CTLA4", "TNFRSF18", "IKZF2"],
        # Th1
        "Th1": ["IFNG", "TBX21", "IL12RB2", "CXCR3", "CCR5"],
        # Th2
        "Th2": ["IL4", "IL5", "IL13", "GATA3", "CCR4"],
        # Th17
        "Th17": ["IL17A", "IL17F", "RORC", "CCR6", "IL23R"],
        # B cells
        "B_general": ["CD19", "CD79A", "CD79B", "MS4A1", "CD22", "PAX5"],
        "B_naive": ["TCL1A", "FCER2", "IL4R"],
        "B_memory": ["CD27", "TNFRSF13B"],
        "Plasma": ["CD38", "SDC1", "XBP1", "PRDM1", "TNFRSF17"],
        # Monocytes
        "Mono_CD14": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1"],
        "Mono_CD16": ["FCGR3A", "CDKN1C", "CX3CR1", "LILRB2"],
        # Dendritic cells
        "mDC": ["ITGAX", "CD1C", "FCER1A", "CLEC10A"],
        "pDC": ["IL3RA", "CLEC4C", "LILRA4", "IRF7", "IRF8"],
        # Granulocytes
        "Neutrophil": ["MPO", "ELANE", "AZU1", "CTSG", "PRTN3", "CXCR2"],
        "Eosinophil": ["IL5RA", "SIGLEC8", "CCR3", "GATA2", "PRG2"],
        "Basophil": ["TPSAB1", "CPA3", "MS4A2", "HDC"],
        # Additional markers
        "Memory_markers": ["SELL", "CCR7", "CD27", "CD45RA", "CD45RO"],
        "Activation": ["CD69", "CD25", "HLA-DR", "CD38", "CD95"],
        "Cytotoxic": ["GZMA", "GZMB", "GZMH", "GZMK", "PRF1", "GNLY"],
    }

    # Flatten all genes and remove duplicates
    all_genes = []
    for genes in signature_genes.values():
        all_genes.extend(genes)
    all_genes = sorted(list(set(all_genes)))

    # Add some additional genes to reach ~317 total
    target_n_genes = 317
    while len(all_genes) < target_n_genes:
        all_genes.append(f"SIG_GENE_{len(all_genes):03d}")

    all_genes = all_genes[:target_n_genes]

    # Create base expression matrix (baseline + noise)
    n_genes = len(all_genes)
    n_cell_types = len(cell_types)

    # Baseline expression (5-7 in log2 scale)
    baseline = np.random.uniform(5, 7, (n_genes, n_cell_types))

    # Add gaussian noise
    noise = np.random.randn(n_genes, n_cell_types) * 0.5
    matrix = baseline + noise

    # Create gene to cell type mapping for specific expression
    gene_to_cells = {}

    # Map NK markers to NK cell types
    for gene in signature_genes["NK"]:
        if gene in all_genes:
            gene_to_cells[gene] = [
                "CD56bright_natural_killer_cell",
                "CD56dim_natural_killer_cell",
            ]

    # Map T cell markers
    for gene in signature_genes["T_general"]:
        if gene in all_genes:
            gene_to_cells[gene] = [
                ct for ct in cell_types if "T_cell" in ct or "T" in ct.split("_")[-1]
            ]

    for gene in signature_genes["CD8_T"]:
        if gene in all_genes:
            gene_to_cells[gene] = [
                "CD8_positive_alpha_beta_T_cell",
                "CD8_Tcm",
                "CD8_Tem",
            ]

    for gene in signature_genes["CD4_T"]:
        if gene in all_genes:
            gene_to_cells[gene] = [
                "CD4_positive_alpha_beta_T_cell",
                "CD4_Tcm",
                "CD4_Tem",
                "Treg",
                "Th1",
                "Th2",
                "Th17",
                "Tfh",
            ]

    for gene in signature_genes["Treg"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["Treg"]

    for gene in signature_genes["Th1"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["Th1"]

    for gene in signature_genes["Th2"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["Th2"]

    for gene in signature_genes["Th17"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["Th17"]

    # Map B cell markers
    for gene in signature_genes["B_general"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["naive_B_cell", "memory_B_cell", "plasma_cell"]

    for gene in signature_genes["B_naive"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["naive_B_cell"]

    for gene in signature_genes["B_memory"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["memory_B_cell"]

    for gene in signature_genes["Plasma"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["plasma_cell"]

    # Map monocyte markers
    for gene in signature_genes["Mono_CD14"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["CD14_positive_monocyte"]

    for gene in signature_genes["Mono_CD16"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["CD16_positive_monocyte"]

    # Map DC markers
    for gene in signature_genes["mDC"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["myeloid_dendritic_cell"]

    for gene in signature_genes["pDC"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["plasmacytoid_dendritic_cell"]

    # Map granulocyte markers
    for gene in signature_genes["Neutrophil"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["neutrophil"]

    for gene in signature_genes["Eosinophil"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["eosinophil"]

    for gene in signature_genes["Basophil"]:
        if gene in all_genes:
            gene_to_cells[gene] = ["basophil"]

    # Apply cell-type specific expression boosts
    for gene, target_cells in gene_to_cells.items():
        if gene in all_genes:
            gene_idx = all_genes.index(gene)
            for cell_type in target_cells:
                if cell_type in cell_types:
                    cell_idx = cell_types.index(cell_type)
                    # Boost expression by 2-4 log2 units
                    boost = np.random.uniform(2, 4)
                    matrix[gene_idx, cell_idx] += boost

    # Create DataFrame
    df = pd.DataFrame(matrix, index=all_genes, columns=cell_types)

    # Save to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path)
        print(f"Saved immunoStates reference matrix to: {output_path}")
        print(f"  Dimensions: {df.shape[0]} genes × {df.shape[1]} cell types")
        print(f"  Value range: [{df.min().min():.2f}, {df.max().max():.2f}]")

        # Show some statistics
        print("\nExpression statistics:")
        print(df.describe().loc[["mean", "std", "min", "max"]])

        print("\nSample genes:")
        print(df.head(10))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate realistic immunoStates reference matrix")
    parser.add_argument(
        "--output",
        "-o",
        default="immunoStates_reference_matrix.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Generating Realistic ImmunoStates Reference Matrix")
    print("=" * 70)

    matrix = create_realistic_immunostates_matrix(output_path=args.output, random_seed=args.seed)

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print("\nYou can now use this matrix for:")
    print("  - Testing ReferenceDataManager")
    print("  - Development and validation")
    print("  - Educational demonstrations")
    print("\nLoad it with:")
    print("  from metaintegrator import ImmunoStatesDeconvolver")
    print(f"  deconvolver = ImmunoStatesDeconvolver(basis_matrix_path='{args.output}')")
