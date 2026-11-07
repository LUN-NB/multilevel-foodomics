# Multilevel Foodomics

## Overview

Multilevel Foodomics provides research code and frozen data for analysing compound–protein target profiles in multicomponent foods. **CPI-MEC** integrates docking scores, auxiliary predictions and experimental interaction labels to calibrate compound–protein scores. **Atlas-Structure** diagnoses the structure of the resulting continuous profiles, identifies MacroGroups and partitions each group into subclusters. The supplied Atlas analysis covers 465 honeysuckle compounds.

## Repository structure

- [CPI-MEC](CPI-MEC/) contains Python source, model configurations, training matrices and labels, and cross-validation assignments in [reports/splits](CPI-MEC/reports/splits/). The [results](CPI-MEC/results/) include model selections, strict outer-fold predictions and evaluation tables, and final full-label fits.
- [Atlas-Structure](Atlas-Structure/) contains R scripts, path settings, frozen subcluster score inputs, and structural results. The [final assignments](Atlas-Structure/results/04D/Subcluster_Assignments.csv) link each compound to its MacroGroup and subcluster.

## Installation

Use Git LFS to retrieve the GraphBAN matrix. Run `git lfs install` before cloning, or `git lfs pull` in an existing checkout. Use Python 3.11 with the pinned dependencies and PyTorch 2.7.1. The commands below install the CUDA 11.8 build for compatible Linux and Windows systems.

~~~sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r CPI-MEC/requirements.txt
~~~

For Atlas-Structure, use R 4.5.0 with CRAN packages `cluster`, `clValid`, `igraph`, `irlba`, `mclust`, `Seurat`, `SeuratObject` and `dplyr`. Install `destiny` and `SingleCellExperiment` through Bioconductor and follow the [Monocle 3 installation guide](https://cole-trapnell-lab.github.io/monocle3/docs/installation/) for trajectory analysis.

## Usage

Run these commands from the repository root. The CPI-MEC preflight checks the packaged files and runtime. The second command starts the full workflow and writes to `CPI-MEC/reports/adaptive_pipeline_complete_gpu_trial_5x5/`.

~~~sh
python CPI-MEC/run_pipeline.py --verify-only
python CPI-MEC/run_pipeline.py
~~~

The [model configuration](CPI-MEC/configs/tuning/adaptive_phase2_4gb.yaml) lists the formal inputs. Frozen [preZ](CPI-MEC/results/final_fit/Final_full_label_fit_preZ_matrix.csv) and [rowZ](CPI-MEC/results/final_fit/Final_full_label_fit_rowZ_matrix.csv) matrices are available under `CPI-MEC/results/final_fit/`.

~~~sh
Rscript --vanilla Atlas-Structure/src/03B_structure.R
Rscript --vanilla Atlas-Structure/src/04A_trajectories.R
Rscript --vanilla Atlas-Structure/src/04A_subclusters.R
Rscript --vanilla Atlas-Structure/src/04D_assignments.R
~~~

Atlas uses the frozen CPI-MEC rowZ matrix. The subcluster entry uses the bundled frozen score tables. The final join reads frozen upstream assignments by default. New Atlas outputs are written under `Atlas-Structure/runs/`.

## Citation

Cite **Multilevel Foodomics** and identify the commit used. See [CITATION.cff](CITATION.cff).

Target priors harmonize and combine [SwissTargetPrediction](https://www.swisstargetprediction.ch/termsofuse.php) materials (SIB, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) and [TCMSP](https://www.tcmsp-e.com/) records. Labels draw on [GraphBAN](https://www.nature.com/articles/s41467-025-57536-9), [NPASS 3.0](https://academic.oup.com/nar/article/54/D1/D1519/8324957) and [DrugBAN](https://github.com/peizhenbai/DrugBAN). Identity and annotation resources include [PubChem](https://pubchem.ncbi.nlm.nih.gov/docs/citation-guidelines), [UniProt](https://www.uniprot.org/help/license) and [NPClassifier](https://github.com/mwang87/NP-Classifier#license).
