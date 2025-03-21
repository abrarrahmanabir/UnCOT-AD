"""

This folder contains the processed datasets used in the UnCOT-AD study for Alzheimer's Disease (AD) prediction 
and cross-omics translation.

Each file corresponds to a different omics modality:

1. gene_expression.xlsx
2. dna_methylation.xlsx
3. proteomics.xlsx

File Format:
-------------
Each `.xlsx` file is organized as follows:
- Rows correspond to individual samples.
- Columns (except the last) represent omics features:
    - For gene_expression.xlsx: each column is a gene.
    - For dna_methylation.xlsx: each column is a methylation site.
    - For proteomics.xlsx: each column is a protein.
- The **last column** in each file is a binary label:
    - `0` = Control sample (non-AD)
    - `1` = Alzheimer's Disease (AD) sample

Data Preprocessing:
--------------------
- The gene expression and DNA methylation data were obtained from GSE33000, GSE44770, and GSE80970 respectively, 
  as described in Park et al. (2020).
- The proteomics dataset was obtained from Shokhirev et al. (2022).
- Differentially expressed genes (DEGs), differentially methylated positions (DMPs), and differentially expressed proteins (DEPs)
  were identified using statistical comparison (t-test) between AD and control samples.
- Thresholds used:
    - DEGs: p < 0.01 and |fold change| ≥ 2
    - DMPs: p < 0.01 and |fold change| ≥ 1.5
    - DEPs: p < 0.05 and |fold change| ≥ 1.5

Usage:
-------
These data files are used as inputs for training and evaluating the UnCOT-AD model.

Please cite the original sources if using this data:
- Park et al. (2020)
- Shokhirev et al. (2022)
"""

