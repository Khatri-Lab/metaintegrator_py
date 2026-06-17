# Citation Guide

If you use MetaIntegrator-py in your research, please cite both the software and the original methodology paper.

---

## Citing MetaIntegrator-py (Python Implementation)

### Recommended Citation

> **MetaIntegrator-py: A Python implementation for meta-analysis of gene expression data**  
> Khatri Lab, Stanford University (2026)  
> https://github.com/Khatri-Lab/metaintegrator_py  
> Version 1.0.0

### BibTeX

```bibtex
@software{metaintegrator_py2026,
  title = {MetaIntegrator-py: A Python implementation for meta-analysis of gene expression data},
  author = {{Khatri Lab}},
  year = {2026},
  url = {https://github.com/Khatri-Lab/metaintegrator_py},
  version = {1.0.0},
  institution = {Stanford University},
  note = {Python package version 1.0.0}
}
```

### RIS Format

```
TY  - COMP
TI  - MetaIntegrator-py: A Python implementation for meta-analysis of gene expression data
AU  - Khatri Lab
PY  - 2026
UR  - https://github.com/Khatri-Lab/metaintegrator_py
PB  - Stanford University
ET  - 1.0.0
ER  -
```

---

## Citing the Original Methodology

**You must also cite the original R MetaIntegrator paper that describes the methodology:**

### Primary Citation

> Haynes WA, Vallania F, Liu C, et al. **Empowering multi-cohort gene expression analysis to increase reproducibility.** *Bioinformatics*. 2017;33(4):596-598. [doi:10.1093/bioinformatics/btw694](https://doi.org/10.1093/bioinformatics/btw694)

### BibTeX

```bibtex
@article{haynes2017metaintegrator,
  title = {Empowering multi-cohort gene expression analysis to increase reproducibility},
  author = {Haynes, Winston A and Vallania, Francesco and Liu, Chun and others},
  journal = {Bioinformatics},
  volume = {33},
  number = {4},
  pages = {596--598},
  year = {2017},
  publisher = {Oxford University Press},
  doi = {10.1093/bioinformatics/btw694}
}
```

---

## Citation by Feature Used

If you use specific features, consider citing additional references:

### COCONUT Integration
If you use the COCONUT batch correction functionality:

> Sweeney TE, Haynes WA, Vallania F, Ioannidis JP, Khatri P. **Methods to increase reproducibility in differential gene expression via meta-analysis.** *Nucleic Acids Research*. 2017;45(1):e1. [doi:10.1093/nar/gkw797](https://doi.org/10.1093/nar/gkw797)

### Immune Deconvolution
If you use the ImmunoStates immune cell deconvolution:

> Vallania F, Tam A, Lofgren S, et al. **Leveraging heterogeneity across multiple datasets increases cell-mixture deconvolution accuracy and reduces biological and technical biases.** *Nature Communications*. 2018;9(1):4735. [doi:10.1038/s41467-018-07242-6](https://doi.org/10.1038/s41467-018-07242-6)

### Gene Set Scoring
If you use the gene signature scoring functionality:

> Sweeney TE, Braviak L, Tato CM, Khatri P. **Genome-wide expression for diagnosis of pulmonary tuberculosis: a multicohort analysis.** *Lancet Respiratory Medicine*. 2016;4(3):213-224. [doi:10.1016/S2213-2600(16)00048-5](https://doi.org/10.1016/S2213-2600(16)00048-5)

---

## Example Citations in Text

### Methods Section

> "Meta-analysis was performed using MetaIntegrator-py v0.1.0 (Khatri Lab, Stanford University, 2026) which implements the methodology described by Haynes et al. (2017). Random-effects models were used to pool effect sizes across studies, and leave-one-out analysis assessed result robustness."

### Results Section

> "Using MetaIntegrator-py (https://github.com/Khatri-Lab/metaintegrator_py), we identified 245 differentially expressed genes (FDR < 0.05, |ES| > 0.5) that were consistent across all cohorts."

### Software Availability Statement

> "All analyses were conducted using MetaIntegrator-py v0.1.0, available at https://github.com/Khatri-Lab/metaintegrator_py under Apache 2.0 license. Code to reproduce our analysis is available at [your repository]."

---

## DOI and Archival

### Zenodo DOI (Coming Soon)

A Zenodo DOI will be assigned to each major release for permanent archival and citation:

```bibtex
@software{metaintegrator_py_zenodo,
  title = {MetaIntegrator-py},
  author = {{Khatri Lab}},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},  % DOI will be assigned upon first release
  url = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

**Note**: Check the [GitHub Releases](https://github.com/Khatri-Lab/metaintegrator_py/releases) page for the specific DOI of the version you used.

---

## Citation Formats by Journal

### Nature Journals

> Khatri Lab. MetaIntegrator-py: A Python implementation for meta-analysis of gene expression data. *GitHub* https://github.com/Khatri-Lab/metaintegrator_py (2026).

### APA Style (7th Edition)

> Khatri Lab. (2026). *MetaIntegrator-py: A Python implementation for meta-analysis of gene expression data* (Version 1.0.0) [Computer software]. Stanford University. https://github.com/Khatri-Lab/metaintegrator_py

### Chicago Style

> Khatri Lab. 2026. "MetaIntegrator-py: A Python Implementation for Meta-Analysis of Gene Expression Data." Computer software. Stanford University. https://github.com/Khatri-Lab/metaintegrator_py.

### Vancouver Style

> Khatri Lab. MetaIntegrator-py: A Python implementation for meta-analysis of gene expression data [Internet]. Stanford (CA): Stanford University; 2026 [cited 2026 Apr 22]. Available from: https://github.com/Khatri-Lab/metaintegrator_py

---

## ORCID iDs

When citing, please include ORCID iDs when available:

- **Purvesh Khatri**: [0000-0002-0364-5028](https://orcid.org/0000-0002-0364-5028)

---

## CFF (Citation File Format)

A `CITATION.cff` file is included in the repository root for automated citation generation:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite both the software and the methodology paper."
type: software
title: "MetaIntegrator-py"
version: 1.0.0
date-released: 2026-04-22
url: "https://github.com/Khatri-Lab/metaintegrator_py"
repository-code: "https://github.com/Khatri-Lab/metaintegrator_py"
license: Apache-2.0
authors:
  - family-names: "Khatri Lab"
    affiliation: "Stanford University"
preferred-citation:
  type: article
  authors:
    - family-names: "Haynes"
      given-names: "Winston A"
    - family-names: "Vallania"
      given-names: "Francesco"
    - family-names: "Liu"
      given-names: "Chun"
  title: "Empowering multi-cohort gene expression analysis to increase reproducibility"
  journal: "Bioinformatics"
  year: 2017
  volume: 33
  issue: 4
  start: 596
  end: 598
  doi: "10.1093/bioinformatics/btw694"
```

---

## Questions About Citation?

If you're unsure how to cite MetaIntegrator-py for your specific use case:
- Open an issue: https://github.com/Khatri-Lab/metaintegrator_py/issues
- Email: khatrilab-support@stanford.edu

---

## Thank You!

Citations help demonstrate the impact of open-source scientific software and support continued development. Thank you for citing MetaIntegrator-py in your work!

---

## See Also

- [License Information](license.md)
- [Release Notes](changelog.md)
- [Contributing Guidelines](../contributing/guidelines.md)
