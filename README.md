# LEO-CVAE: An Uncertainty-Aware Approach to Generative Oversampling
[![arXiv](https://img.shields.io/badge/arXiv-2509.25334-b31b1b.svg)](https://arxiv.org/abs/2509.25334)

This repository contains the official implementation for the paper "Uncertainty-Aware Generative Oversampling Using an
Entropy-Guided Conditional Variational Autoencoder".

> **Abstract:** Class imbalance remains a major challenge in machine learning, especially for high-dimensional biomedical data where nonlinear manifold structures dominate. Traditional oversampling methods such as SMOTE rely on local linear interpolation, often producing implausible synthetic samples. Deep generative models like Conditional Variational Autoencoders (CVAEs) better capture nonlinear distributions, but standard variants treat all minority samples equally, neglecting the importance of uncertain, boundary-region examples emphasized by heuristic methods like Borderline-SMOTE and ADASYN.
We propose Local Entropy-Guided Oversampling with a CVAE (LEO-CVAE), a generative oversampling framework that explicitly incorporates local uncertainty into both representation learning and data generation. To quantify uncertainty, we compute Shannon entropy over the class distribution in a sample's neighborhood: high entropy indicates greater class overlap, serving as a proxy for uncertainty. LEO-CVAE leverages this signal through two mechanisms: (i) a Local Entropy-Weighted Loss (LEWL) that emphasizes robust learning in uncertain regions, and (ii) an entropy-guided sampling strategy that concentrates generation in these informative, class-overlapping areas.
Applied to clinical genomics datasets (ADNI and TCGA lung cancer), LEO-CVAE consistently improves classifier performance, outperforming both traditional oversampling and generative baselines. These results highlight the value of uncertainty-aware generative oversampling for imbalanced learning in domains governed by complex nonlinear structures, such as omics data.

## Setup

Clone the repository and install the required packages. A Python 3.12 virtual environment is recommended.

```bash
# Clone the repository
git clone https://github.com/Amirhossein-Zare/LEO-CVAE.git
cd LEO-CVAE

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


## Citation

To cite this work, please use the following BibTeX entry:

```bibtex
@misc{zare2025uncertaintyawaregenerativeoversamplingusing,
      title={Uncertainty-Aware Generative Oversampling Using an Entropy-Guided Conditional Variational Autoencoder}, 
      author={Amirhossein Zare and Amirhessam Zare and Parmida Sadat Pezeshki and Herlock and Rahimi and Ali Ebrahimi and Ignacio Vázquez-García and Leo Anthony Celi},
      year={2025},
      eprint={2509.25334},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.25334}, 
}
