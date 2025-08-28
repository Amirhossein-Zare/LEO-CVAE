# Entropy-Guided CVAE: An Uncertainty-Aware Approach to Generative Oversampling

This repository contains the official implementation for the paper "Uncertainty-Aware Generative Oversampling Using an
Entropy-Guided Conditional Variational Autoencoder".

## Setup

Clone the repository and install the required packages. A Python 3.12 virtual environment is recommended.

```bash
# Clone the repository
git clone [https://github.com/your-username/leo-cvae-project.git](https://github.com/your-username/leo-cvae-project.git)
cd leo-cvae-project

# Create and activate a virtual environment 
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

# Data

Due to data use agreements, the raw TCGA and ADNI datasets used in this study cannot be shared in this repository.  
The scripts in this repository expect the data to be preprocessed as described in the paper.


## 1. Data Acquisition

You must first acquire the data from the official sources:

- TCGA:  
  The TCGA-LUAD and TCGA-LUSC gene expression datasets can be downloaded from the [UCSC Xena Browser](https://xenabrowser.net/).

- ADNI:  
  The ADNI gene expression dataset is available to qualified researchers upon application through the [Laboratory of Neuro Imaging (LONI)](http://adni.loni.usc.edu/).


## 2. Preprocessing

The raw data must be preprocessed before it can be used with `main.py`.  
The key steps, as detailed in our paper, include:

1. Merging the relevant clinical and gene expression data files.  
2. Applying Mutual Information (MI) based feature selection to reduce the feature set to the 64 most informative features.  
3. Defining and encoding the target labels for the classification tasks.  

After performing these steps, place the final, processed data files in the main project directory.




## Usage

Once the data has been preprocessed and placed in the main project directory, the entire 5-fold cross-validation experiment can be run from the main script. Configurations can be modified in config.py.

```bash
python main.py
```
