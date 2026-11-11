# Uncertainty Quantification in Skin Cancer Classification

## Overview

This project implements a Three-Way Decision-Based Bayesian Deep Learning approach for skin cancer classification.

The main objective is not only to classify skin lesions as benign or malignant, but also to estimate the uncertainty of model predictions. This helps identify uncertain cases that may require further medical evaluation.

The implementation is based on the research paper:

**Uncertainty Quantification in Skin Cancer Classification Using Three-Way Decision-Based Bayesian Deep Learning**

Moloud Abdar et al., *Computers in Biology and Medicine*, 2021.

DOI: https://doi.org/10.1016/j.compbiomed.2021.104418

## Objectives

- Classify skin lesions using deep learning.
- Estimate predictive uncertainty.
- Implement Bayesian uncertainty estimation techniques.
- Apply Three-Way Decision (TWD) theory.
- Defer uncertain predictions for further evaluation.
- Compare model performance using classification metrics.

## Methodology

### 1. Deep Learning Architectures

The implementation uses four pretrained ImageNet architectures:

- DenseNet201
- MobileNetV2
- ResNet152V2
- InceptionResNetV2

The pretrained backbones are frozen, with additional fully connected layers and dropout for binary classification.

### 2. Uncertainty Quantification

Three uncertainty estimation techniques are implemented:

- **Monte Carlo Dropout (MC Dropout):** Performs multiple stochastic forward passes to estimate predictive uncertainty.
- **Deep Ensemble (DE):** Combines predictions from independently trained models.
- **Ensemble Monte Carlo Dropout (EMC):** Combines ensemble learning with Monte Carlo Dropout.

### 3. Three-Way Decision Theory

Predictions are organized into three possible decisions:

- Accept: High-confidence predictions.
- Defer: Uncertain predictions requiring further evaluation.
- Reject: Predictions outside the accepted decision region.

The notebook implements a multi-phase referral pipeline.

## Datasets

The implementation supports two binary skin cancer datasets:

1. Skin Cancer: Malignant vs. Benign.
2. ISIC 2019 binary melanoma / non-melanoma preparation.

The datasets must be supplied separately through Kaggle with the folder structures expected by the notebook.

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- SciPy
- Jupyter Notebook
- Kaggle

## Evaluation Metrics

The implementation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Predictive Entropy
- Predictive Standard Deviation

It also includes Grad-CAM visualizations, uncertainty plots, and training performance curves.

## Repository Structure

```text
skincancerbayesian/
│
├── README.md
└── probability.ipynb
```

## How to Run

1. Clone or download this repository.
2. Open the notebook in Kaggle or a compatible Jupyter environment.
3. Upload or attach the required datasets.
4. Install the required dependencies if they are not already available.
5. Execute the notebook cells sequentially.

**Note:** The full experimental pipeline is computationally intensive. A Kaggle GPU runtime is recommended.

## Research and Reproducibility

This notebook attempts to reproduce the main experimental pipeline described in the research paper.

Some implementation details, including exact uncertainty thresholds and Bayesian optimization settings, are not fully specified in the original paper. These limitations are documented in the notebook.

## Disclaimer

This project is intended for academic and research purposes only. It is not a clinically validated diagnostic system and should not be used as a substitute for professional medical evaluation.

## Author

**Kasturi Yaswanth Krishna**

GitHub: https://github.com/kasturiyaswanthk-cyber
