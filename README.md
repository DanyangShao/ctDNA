# ctDNA-CancerNet

Transfer learning framework for multi-cancer classification using circulating tumor DNA (ctDNA) methylation profiles.

## Overview

This repository contains the code and processed data used in the manuscript:

**"Cancer detection and classification using circulating tumor DNA from blood samples"**

The proposed framework, **ctDNA-CancerNet**, integrates:

- Variational Autoencoders (VAEs)
- Transfer learning from CancerNet
- ctDNA methylation profiling
- SHAP-based post-hoc interpretability analysis

to perform multi-class cancer classification using heterogeneous ctDNA methylation datasets.

---

## Included Cancer Types

The current study includes the following classes:

- NSCLC (Non-Small Cell Lung Cancer)
- CRC (Colorectal Cancer)
- BRCA (Breast Cancer)
- MCC (Merkel Cell Carcinoma)
- PRAD (Prostate Adenocarcinoma)
- HCC (Hepatocellular Carcinoma)
- Normal controls

---

## Data Sources

The ctDNA methylation datasets were obtained from:

- GEO:
  - GSE243474
  - GSE40279
  - GSE157273
- RDDB:
  - RDDB2017000132

TCGA methylation data were used for CancerNet pretraining.

---

## Preprocessing Pipeline

The preprocessing workflow includes:

1. Data harmonization across heterogeneous cohorts
2. Quantile normalization of methylation beta values
3. CpG clustering using CancerLocator
4. Removal of samples containing empty clustered feature values
5. Construction of 24,565 CpG-cluster features from 473,034 CpG sites

---

## Model Architecture

The ctDNA-CancerNet framework uses:

- Variational Autoencoder (VAE)
- Transfer learning from pretrained CancerNet encoder
- Two-stage fine-tuning strategy

### Architecture Details

- Encoder hidden layers: 1000 → 500
- Latent dimension: 100
- Classifier hidden layer: 100
- Output classes: 7

---

## Transfer Learning Strategy

Stage 1:
- Selected pretrained encoder feature extraction layers are frozen
- Downstream latent and classification layers remain trainable

Stage 2:
- Full network fine-tuning with reduced learning rate

Optimization:
- Adam optimizer
- Learning rates:
  - Stage 1: 1e-4
  - Stage 2: 1e-5
- Early stopping based on validation loss
- Patience = 120 epochs

---

## Cross Validation

Evaluation was performed using:

- Stratified 5-fold cross-validation
- Approximate 60:20:20 train-validation-test partition per fold

Reported metrics include:

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1

---

## Interpretability

Model interpretability was performed using:

- SHAP latent-space attribution
- Integrated Gradients
- CpG-to-gene mapping using Illumina HumanMethylation450K annotations

SHAP was used solely as a post-hoc interpretability framework and not for feature selection.

---


Example environment:

Python 3.10+
TensorFlow 2.x
NumPy
pandas
scikit-learn
matplotlib
shap
bedtools

Install dependencies:

pip install -r requirements.txt
Reproducibility

To improve reproducibility:

Random seed = 42
Stratified cross-validation used across all experiments
All reported results are averaged across 5 folds
Limitations

This study represents a proof-of-concept framework using heterogeneous public ctDNA methylation datasets. Additional external validation and larger independent cohorts are needed before clinical translation.

Citation

If you use this repository, please cite:

Shao D, Azad RK.
Cancer detection and classification using circulating tumor DNA from blood samples.
Contact

Danyang Shao
University of North Texas

Rajeev K. Azad
University of North Texas
