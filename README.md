# Code supplement to the CNV convergence
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F862615-informational
)]([https://doi.org/10.1101/862615](https://doi.org/10.1101/2022.04.23.489093))

This repository contains the code used to process and analyse the data presented in the "Rare CNVs and phenome-wide profiling highlight brain-structural divergence and phenotypical convergence" paper. 

## Abstract

<div style="margin-left: 40px;" align="justify">
Copy number variations (CNVs) are rare genomic deletions and duplications that can affect brain and behavior. Previous reports of CNV pleiotropy imply that they converge on shared mechanisms at some level of pathway cascades, from genes to large-scale neural circuits to the phenome. However, existing studies have primarily examined single CNV loci in small clinical cohorts. It remains unknown how distinct CNVs escalate vulnerability for the same developmental and psychiatric disorders. Here, we quantitatively dissect the associations between brain organization and behavioral differentiation across eight key CNVs. In 534 CNV carriers, we explored CNV-specific brain morphology patterns. CNVs were characteristic of disparate morphological changes involving multiple large-scale networks. We extensively annotated these CNV-associated patterns with ~1000 lifestyle indicators through the UK Biobank resource. The resulting phenotypic profiles largely overlap and have body-wide implications, including the cardiovascular, endocrine, skeletal, and nervous systems. Our population-level investigation established brain structural divergences and phenotypical convergences of CNVs, with direct relevance to major brain disorders.
</div>


<c>![Figure 1](https://github.com/jakubkopal/CNV-convergence/blob/main/Figures/Fig1.png)</c>


## Resources and Scripts
LDA classification accuracies are saved in the `Data/CNV_IP` folder. Phenotypic associations are saved in the `Data/PheWAS` folder. Findings from the article are based on the analysis scripts in the `Scripts` folder.

1.   `Scripts/calculate_ip.py` is an example of using a Linear discriminant to isolate CNV-specific intermediate phenotypes from raw anatomical data. Bagging and regularization are implemented to safeguard before overfitting.
2.   `Scripts/lowdim_projection.py` is an analysis script to plot PCA and LDA two-dimensional projection of regional volumes.
3.   `Scripts/compare_cohen_ip.py` plots and compares derived CNV-specific Cohen's d brain maps with LDA-derived intermediate phenotypes.
4.   `Scripts/analyze_ip.py` is an analysis script to investigate CNV effects on large-scale networks with the use of LDA intermediate phenotypes.
5.   `Scripts/compare_simmilarities.py` is an analysis script to compute the similarity between the phenome profiles, intermediate phenotypes, and volumetric Cohen's d brain maps.
