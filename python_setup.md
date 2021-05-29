# Python code set up

These are instructions for installing Python and its libraries that are needed for making plots of the simulation output.

## Install Miniconda

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). It is used for creating isolated Python environments that use specific version of Python and the libraries for this project (listed in [requirements.txt](requirements.txt)). This also avoids polluting system-wide python installation and interfering with other Python projects.


## Create Conda environment

Create a Conda environment with Python 3.8:

```
conda create --name uni_2021_data_analysis_problem_sets python=3.8
```

Answer YES to when asked "The following NEW packages will be INSTALLED".


## Activate Conda environment

```
conda activate uni_2021_data_analysis_problem_sets
```

## Install Python libraries

Follow the steps from [README](README.md) to download the project and make sure you are in the working directory:

```
cd uni_2021_data_analysis_problem_sets
```

Install Python libraries listed in [requirements.txt](requirements.txt) file:

```
pip install -r requirements.txt
```

It will install the exact versions of Python libraries that were used by Python programs in this project. Now you are ready to run Python code.


## Clean up

To remove the `uni_2021_data_analysis_problem_sets` conda environment completely and free up disk space:

```
conda deactivate
conda env remove -n uni_2021_data_analysis_problem_sets
```
