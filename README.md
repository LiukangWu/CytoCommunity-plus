# CytoCommunity+

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Update Log](#update-log)
- [Maintainers](#maintainers)
- [Citation](#citation)



## Overview

To enhance CytoCommunity (https://github.com/huBioinfo/CytoCommunity), we present CytoCommunity+, **a unified weakly-supervised framework** for identifying and comparing tissue cellular neighborhoods (TCNs or CNs) across large-scale spatial omics samples with single or multiple biological conditions. 

Inspired by histopathology workflows, CytoCommunity+ first hierarchically partitions the large single-cell spatial map into small patches, performs graph construction and weakly supervised TCN learning for each patch, and finally merges results through KNN-based TCN reassignment at segmentation boundaries to ensure TCN spatial continuity. This strategy divides the original sample into patches for TCN learning, achieving **memory efficiency (typical 24G graphics memory is enough)** and also increased sample throughput. These improvements facilitate robust GNN-based TCN learning and cross-sample alignment. 

Furthermore, to make CytoCommunity+ a unified framework that is also suitable for spatial omics datasets with a single condition, pseudo-samples with artificial labels are generated, enabling automatic TCN alignment across real samples via contrastive learning.

## Installation

### Hardware requirement 

CPU: i7

Memory: 16G or more

Storage: 10GB or more

### Software requirement

Conda version: 22.9.0

Python version: 3.10.6

R version: >= 4.0 suggested

Clone this repository and cd into it as below.
```
git clone https://github.com/LiukangWu/CytoCommunity-plus.git
cd CytoCommunity-plus
```
#### For Windows

#### Preparing the virtual environment

1. Create a new conda environment using the environment_windows_cpu.yml file (CPU version) and activate it:

    ```bash
    conda env create -f environment_windows_cpu.yml
    conda activate CytoCommunity_cpu
    ```

    Or create a new conda environment using the environment_windows_gpu.yml file (GPU version) and activate it:

    ```bash
    conda env create -f environment_windows_gpu.yml
    conda activate CytoCommunity_gpu
    ```


2. Install the diceR package (R has already been included in the requirements) with the following command:

    ```bash
    R.exe
    > install.packages("diceR")
    ```

#### For Linux

#### Preparing the virtual environment 

1. Create a new conda environment using the environment_linux_cpu.yml file (CPU version) and activate it:

    ```bash
    conda env create -f environment_linux_cpu.yml
    conda activate CytoCommunity_cpu
    ```

    Or create a new conda environment using the environment_linux_gpu.yml file (GPU version) and activate it:

    ```bash
    conda env create -f environment_linux_gpu.yml
    conda activate CytoCommunity_gpu
    ```


2. Install R and the diceR package:
    
    ```bash
    conda install R
    R
    > install.packages("diceR")
    ```

The whole installation should take around 20 minutes.


## Usage

#### Prepare input data

The input data to CytoCommunity+ includes **four types of files: 

(1) An image (sample) name list file, named as **"ImageNameList.txt"**.

(2) A cell type label file for each image (sample), named as **"[image name]_CellTypeLabel.txt"**. Note that [image_name] should be consistent with your customized image names listed in the "ImageNameList.txt".
This file lists cell type names of all cells in an image (sample).

(3) A cell spatial coordinate file for each image (sample), named as **"[image name]_Coordinates.txt"**. Note that [image_name] should be consistent with your customized image names listed in the "ImageNameList.txt".
This file lists cell coordinates (tab-delimited x/y) of all cells in an image (sample). The cell orders should be exactly the same with "[image name]_CellTypeLabel.txt".

(4) **(Optional, for multi-condition datasets only)** A graph label file for each image (sample), named as **"[image name]_GraphLabel.txt"**. Note that [image_name] should be consistent with your customized image names listed in the "ImageNameList.txt".
This file contains an integer label (e.g., "0", "1", "2", etc) that indicates the condition of each image (sample) in the weakly-supervised learning framework.

Example input files can be found under the directory "CODEX_SpleenDataset/".


#### Run the following steps in Windows Powershell or Linux Bash shell:

#### 0. (optional, for single-condition datasets only) Use Step0 to generate pseudo-spatial maps by shuffling cell types in real spatial maps

This step generates a folder "Step0_Output" containing pseudo-spatial maps created by randomly shuffling cell type labels while maintaining original spatial coordinates. Each pseudo-sample will have corresponding "pseudo" prefixed files alongside the original samples.

```bash
python Step0_GeneratePseudoMaps.py
```
&ensp;&ensp;**Hyperparameters**
- InputFolderName: The folder name of your input dataset (must contain original spatial maps).


#### 1. Use Step1 to construct cellular spatial graphs and split large images into manageable patches
This step generates a folder "Step1_Output" containing spatially split patches for each original image along with their corresponding coordinate files, cell type label files, and graph label files, as well as a global "All_Boundary.txt" file that records all splitting boundaries and an "ImagePatchNameList.txt" file that catalogs all generated patches. The recursive splitting process ensures large tissue images are divided into smaller, more manageable patches while maintaining all original cellular information and spatial relationships.

```bash
python Step1_SplitSpatialMaps.py
```
&ensp;&ensp;**Hyperparameters**
- CellPatchNum: Maximum cell count threshold (default=50,000) triggering recursive splitting
- MinCellCount_Patch: Minimum cell count (default=20) required to keep a generated patch
- InputFolderName: Path to input dataset folder (default="./Step0_Output/")

#### 2. Use Step2 to construct KNN-based cellular spatial graghs and convert the input data to the standard format required by Torch.

This step generates a folder "Step2_Output" including constructed cellular spatial graphs of all samples/images in your input dataset folder (e.g., /CODEX_SpleenDataset/).

```bash
python Step2_ConstructCellularSpatialGraphs.py
```
&ensp;&ensp;**Hyperparameters**
- InputFolderName: The folder name of your input dataset.
- KNN_K: The K value used in the construction of the K nearest neighbor graph (cellular spatial graph) for each sample/image. This value can be empirically set to the integer closest to the square root of the average number of cells in the images in your dataset.

#### 3. Use Step3 to perform soft TCN assignment learning in a weak-supervised fashion.

This step generates a folder "Step3_Output" containing results from multiple independent runs of the supervised TCN learning process. Each run folder includes training loss logs and output matrices (cluster assignment matrix, cluster adjacency matrix, and node mask) for all samples. The model combines graph partitioning (MinCut loss) with graph classification (cross-entropy loss) in an end-to-end training framework.

```bash
python Step3_TCN-Learning_WeaklySupervised.py
```
&ensp;&ensp;**Hyperparameters**
- Num_TCN: Maximum number of TCNs (default=4) to identify
- Num_Run: Number of independent training runs (default=10)
- Num_Epoch: Training epochs per run (default=400)
- Num_Class: Number of graph classes (default=2)
- Embedding_Dimension: Feature dimension (default=128)
- MiniBatchSize: This value is commonly set to be powers of 2 due to efficiency consideration. In the meantime, this value is suggested to be closest to half sizes of the training sets. (default=2)
- LearningRate: Optimizer learning rate (default=0.001)
- beta: A weight parameter to balance the MinCut loss used for graph partitioning and the cross-entropy loss used for graph classification. The default value is set to [0.9] due to emphasis on graph partitioning.

#### 4. Use Step4 to perform TCN assignment ensemble.

The results of this step are saved under the "Step4_Output/ImageCollection/" directory. A "TCNLabel_MajorityVoting.csv" file will be generated for each image. Make sure that the diceR package has been installed before Step4.

```bash
Rscript Step4_TCN-Ensemble.R
```
&ensp;&ensp;**Hyperparameters**
- InputFolderName: The folder name of your input dataset, consistent with Step1.

#### 5. Use Step5 to refine TCN boundaries and generate final visualizations

This step generates a folder "Step5_Output" containing four subfolders with comprehensive results: "TCN_Plot" stores spatial maps colored by identified TCNs (in PNG and PDF formats), "CellRefinement_Plot" shows boundary refinement results, "ResultTable_File" contains detailed spatial data in CSV format, and "CellType_Plot" stores spatial maps colored by original cell type annotations. The step performs boundary smoothing using KNN-based label propagation to improve TCN continuity across patch boundaries.

```bash
python Step5_BoundaryRefinement.py
```
&ensp;&ensp;**Hyperparameters**
- KNN_K: Number of nearest neighbors (default=50) used for boundary refinement
- Num_TCN: Maximum number of TCNs (default=4) for consistent coloring
- Smoothing_range: Spatial range (default=50μm) for boundary refinement
- InputFolderName: Path to input dataset folder (default="./Step0_Output/")



## Update Log


## Maintainers
Liukang Wu (yetong@stu.xidian.edu.cn)

Yafei Xu (22031212416@stu.xidian.edu.cn)

Yuxuan Hu (huyuxuan@xidian.edu.cn)


## Citation
