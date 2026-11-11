# Uncertainty-Aware Skin Lesion Classification
### Three-Way Decision-Based Bayesian Deep Learning

A research-oriented implementation of uncertainty quantification (UQ) for binary skin-lesion classification. The project combines ImageNet-pretrained convolutional neural networks with Monte Carlo (MC) Dropout, Deep Ensembles (DE), Ensemble MC Dropout (EMC), and a three-way decision (TWD) strategy that can defer uncertain predictions instead of forcing an automatic classification.

> **Research/educational project only.** This repository is not a clinically validated diagnostic system. It must not be used to diagnose skin cancer or make medical decisions.

---

## 1. Research reference

This implementation is based on:

Abdar, M. et al. (2021). *Uncertainty quantification in skin cancer classification using three-way decision-based Bayesian deep learning*. **Computers in Biology and Medicine**.

- DOI: https://doi.org/10.1016/j.compbiomed.2021.104418

The notebook documents which settings are reported by the paper and which implementation choices are configurable because the paper does not provide enough detail for exact reproduction.

## 2. Problem statement

A conventional classifier returns a class prediction even when it is uncertain. In a medical-imaging context, an uncertain prediction should be treated carefully. This project explores a pipeline that estimates predictive uncertainty and uses a three-way decision process:

- **Accept:** retain a sufficiently certain prediction.
- **Defer/refer:** send an uncertain case to another decision stage.
- **Final referral:** cases that remain uncertain after the later stage are not included in the automatic final-phase metrics.

The goal is to study the relationship between classification performance, uncertainty, and referral—not to replace clinical assessment.

## 3. Methodology

### 3.1 CNN feature extractors

The notebook uses four ImageNet-pretrained architectures as frozen feature extractors:

- DenseNet201
- MobileNetV2
- ResNet152V2
- InceptionResNetV2

A trainable classification head is added to each backbone. It contains two fully connected layers, dropout (rate 0.5), and a two-class softmax output. The notebook uses categorical cross-entropy and early stopping.

### 3.2 Uncertainty quantification

The implementation includes three UQ approaches:

| Method | Overview |
|---|---|
| MC Dropout | Runs stochastic forward passes with dropout active and aggregates the predictions. The notebook uses 50 passes for the paper-style experiment. |
| Deep Ensemble (DE) | Combines predictions from independently trained model members. |
| Ensemble MC Dropout (EMC) | Combines stochastic predictions from multiple ensemble members. |

The notebook reports predictive entropy and predictive standard deviation. It also distinguishes conventional binary predictive entropy from the sample-entropy calculation used to preserve the scale of the paper's reported tables.

### 3.3 Three-way decision pipeline

The test set passes through three decision phases:

1. **Phase 1:** apply the first uncertainty-aware model/ensemble. Retain sufficiently certain predictions and refer uncertain samples.
2. **Phase 2:** evaluate the referred samples with the next model/ensemble. Retain sufficiently certain predictions and defer the remaining uncertain samples.
3. **Final phase:** combine the automatically retained predictions. Samples still uncertain after Phase 2 remain referred and are excluded from automatic final-phase metrics.

The paper does not publish the exact numerical entropy thresholds. The notebook therefore calibrates referral counts to the paper-reported counts; it does **not** claim to recover the paper's original threshold values.

### 3.4 Analysis and visualizations

The notebook includes:

- Training loss and accuracy curves
- Phase-wise and final result tables
- ROC curves and AUC calculations
- Confusion matrices and classification metrics
- Predictive-distribution, entropy, and standard-deviation plots
- Grad-CAM visualizations
- Comparisons between uncertainty and correct/incorrect predictions
- Paper-reference tables alongside results computed by the notebook

## 4. Datasets

The notebook is configured to discover two folder-structured JPG/JPEG datasets under `/kaggle/input`.

### Dataset 1 — Skin Cancer: Malignant vs. Benign

Expected folder structure:

```text
dataset-1/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

Paper-reported counts: 2,637 training images and 660 test images.

### Dataset 2 — Binary ISIC-2019 preparation

Expected folder structure:

```text
dataset-2/
└── data/
    ├── train/
    │   ├── melanoma/
    │   └── not_melanoma/
    └── test/
        ├── melanoma/
        └── not_melanoma/
```

Paper-reported counts: 7,234 training images and 1,808 test images.

**Dataset access:** obtain the datasets from their respective authorized Kaggle/public sources and comply with their licenses and terms. Attach or place them in the Kaggle input area using the folder structures above. The notebook also provides root-override configuration variables if your directory layout differs.

**Class mappings used by the notebook:**

- Dataset 1: benign = `0`, malignant = `1`
- Dataset 2: melanoma = `0`, not_melanoma = `1`

Check that these mappings match your intended interpretation before evaluating results.

## 5. Repository structure

```text
.
├── skin_cancer_uncertainty.ipynb   # Main experiment and analysis
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

Training artifacts are written by default to:

```text
/kaggle/working/twdbdl_artifacts
```

This is a Kaggle runtime path. If you run the notebook outside Kaggle, change `ARTIFACT_ROOT` in the configuration cell to a writable local directory.

## 6. Environment and installation

### Recommended environment

- Python 3.10 or 3.11
- Jupyter Notebook / JupyterLab, or Kaggle Notebooks
- GPU recommended for the full experiment
- Sufficient disk space for datasets, pretrained weights, and model artifacts

TensorFlow may install differently depending on your operating system, Python version, and GPU/CUDA setup. Use the official TensorFlow installation instructions for platform-specific GPU configuration.

### Install dependencies

```bash
python -m venv .venv
```

Activate the environment:

**Windows (PowerShell)**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open `skin_cancer_uncertainty.ipynb`.

For Kaggle, attach the datasets to the notebook and use the preinstalled TensorFlow environment where possible. Avoid reinstalling TensorFlow unnecessarily in a working Kaggle session.

## 7. Running the notebook

Run the notebook cells sequentially, from top to bottom.

1. Attach both datasets or configure the dataset root overrides.
2. Confirm that TensorFlow detects the intended compute device.
3. Start with a small smoke test by setting `MAX_TRAIN_SAMPLES`, `MAX_TEST_SAMPLES`, and `EPOCHS_OVERRIDE` to small values.
4. Confirm that data discovery, model construction, training, uncertainty estimation, and evaluation complete successfully.
5. Restore the limits to `None` for the full dataset run.
6. Keep `MC_PASSES=50` for the paper-style MC Dropout experiment.
7. Review the generated tables and plots, and distinguish notebook-computed values from paper-reference values.

The full run is computationally expensive, particularly for the larger CNN backbones and EMC stages. Optional baseline sweeps and Bayesian Optimization are disabled by default. Enable `RUN_BASELINE_SWEEPS` or `RUN_BO_SEARCH` only when you have sufficient compute and time.

## 8. Configuration

The main configuration variables are defined near the beginning of the notebook.

| Variable | Purpose |
|---|---|
| `SEED` | Random seed used by Python, NumPy, and TensorFlow |
| `IMAGE_SIZE` | Input image dimensions |
| `BATCH_SIZE` | Training/inference batch size |
| `VAL_SPLIT` | Validation fraction (configurable; paper detail not specified) |
| `DROPOUT_RATE` | Dropout probability |
| `MC_PASSES` | Number of stochastic MC Dropout passes |
| `FULL_RUN` | Full experiment control |
| `MAX_TRAIN_SAMPLES` | Optional training sample limit for smoke tests |
| `MAX_TEST_SAMPLES` | Optional test sample limit for smoke tests |
| `EPOCHS_OVERRIDE` | Optional epoch limit for smoke tests |
| `RUN_BASELINE_SWEEPS` | Enable additional baseline experiments |
| `RUN_BO_SEARCH` | Enable the approximate Bayesian Optimization example |
| `DATASET1_ROOT_OVERRIDE` | Optional Dataset 1 root path |
| `DATASET2_ROOT_OVERRIDE` | Optional Dataset 2 root path |

## 9. Results and reproducibility

The notebook contains paper-reported reference values and computes separate reproduction results from the current run. Do not present the paper's reference metrics as results achieved by this implementation.

The paper reports the following final results:

| Dataset | Accuracy | F1-score | AUC |
|---|---:|---:|---:|
| Dataset 1 | 88.95% | 89.00% | 0.92 |
| Dataset 2 | 90.96% | 91.00% | 0.97 |

These are **paper-reported reference values**, not a guarantee of the results produced by this notebook. Add your own run's metrics only after executing the notebook and verifying the generated output.

### Known reproduction limitations

Some details needed for exact replication are not fully specified in the paper. The notebook makes these choices explicit:

- Validation split: 20% stratified split.
- Preprocessing: ImageNet preprocessing.
- Entropy thresholds: calibrated to reported referral counts rather than copied from published numeric thresholds.
- Bayesian Optimization: optional approximate search; the exact paper search space and trial configuration are not published.
- Grad-CAM: ResNet152V2 is used by default; exact paper figure sample IDs/backbone are not specified.
- Network morphism: the paper mentions it but does not provide a concrete procedure, so the notebook does not invent one.

For meaningful comparisons, record the dataset version, runtime, TensorFlow version, random seed, configuration values, and the metrics generated by your run.

## 10. Responsible use

Skin-lesion classification is a high-stakes medical application. This notebook is a research and learning implementation only. It has not been established as safe or clinically validated for patient care. Model outputs can be wrong, uncertain, biased, or poorly calibrated across populations and imaging conditions. Do not use this project to diagnose, triage, or make treatment decisions. Clinical decisions must be made by qualified healthcare professionals using appropriate validated processes.

## 11. Future improvements

Potential extensions include:

- External validation on an independent dataset
- Calibration analysis (for example, reliability diagrams and expected calibration error)
- Confidence intervals and repeated-seed evaluation
- Subgroup and distribution-shift analysis
- Model-card documentation and dataset provenance tracking
- Reproducible experiment tracking and saved configuration files

## 12. Citation

If you use this implementation in academic work, cite the original paper:

```text
Abdar, M. et al. (2021). Uncertainty quantification in skin cancer
classification using three-way decision-based Bayesian deep learning.
Computers in Biology and Medicine.
https://doi.org/10.1016/j.compbiomed.2021.104418
```

---

**Repository contents:** main Jupyter notebook, project documentation, and Python dependency list.
