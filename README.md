# CytoCommunity-plus 

## Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Update Log](#update-log)
- [Maintainers](#maintainers)
- [Citation](#citation)



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
git clone https://github.com/huBioinfo/CytoCommunity-plus.git
cd CytoCommunity
```
#### For Windows

#### Preparing the virtual environment

1. Create a new conda environment using the environment.yml file or the requirements.txt file with one of the following commands:

    ```bash
    conda env create -f environment.yml
    ```

    Or create a new conda environment using the environment_windows_gpu.yml file with the following commands:

    ```bash
    conda env create -f environment_windows_gpu.yml
    ```

Note that the command should be executed in the directory containing the environment.yml file. And if you use the .txt file, please convert it to the UTF-8 format.


2. Install the diceR package (R has already been included in the requirements) with the following command:

    ```bash
    R.exe
    > install.packages("diceR")
    ```

#### For Linux

#### Preparing the virtual environment 

1. Create a new conda environment using the environment_linux.yml file and activate it:

    ```bash
    conda env create -f environment_linux.yml
    conda activate CytoCommunity
    ```

    Or create a new conda environment using the environment_linux_gpu.yml file with the following commands:

    ```bash
    conda env create -f environment_linux_gpu.yml
    ```


2. Install R and the diceR package:
    
    ```bash
    conda install R
    R
    > install.packages("diceR")
    ```

The whole installation should take around 20 minutes.