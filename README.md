# Masters Project: Parkinson's Disease Classification (Reproducibility Build)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-green)

This repository contains the code and resources for a Master's Project focused on comparing various Machine Learning (ML) techniques applied to a Parkinson's Disease dataset. 

**This repo has been extensively refactored to comply with the Machine Learning Reproducibility Checklist (Pineau v2.0) and the TRIPOD+AI clinical reporting standards.**

## 📊 Dataset

The dataset (`data/raw/dataset.csv`) includes various features derived from movement and motor skill tests (e.g., Duration, AreaError, TimeTriangles, Distance, ZeroVel, ZeroAcc). These features are used to classify individuals as either having Parkinson's Disease or being healthy controls.

See [DATA.md](docs/DATA.md) and [DATASHEET.md](docs/DATASHEET.md) for extended details.

## 📁 Repository Structure

```text
parkinsons-ml-repro/
├── README.md                  # overview + exact reproduce commands + results table
├── LICENSE                    # MIT
├── requirements.txt           # dependencies
├── Dockerfile                 # container setup
├── docs/
│   ├── DATA.md                # dataset source, version, access procedure
│   ├── DATASHEET.md           # Gebru et al. datasheet for the dataset
│   ├── MODEL_CARD.md          # Mitchell et al. model card
│   ├── TRIPOD-AI_checklist.md # Layer B clinical reporting
│   └── REPRODUCIBILITY.md     # Layer A reproducibility 
├── data/
│   ├── raw/                   # gitignored, contains dataset.csv
│   └── processed/             # cleaned features and labels
├── splits/                    # saved train/val/test indices + the seed
├── src/
│   ├── utils.py               # set_seed() for numpy / framework / split
│   ├── data_prep.py           # data exclusions and cleaning
│   ├── features.py            # standardization and explicit dataset splits
│   ├── train.py               # grid search and model saving
│   └── evaluate.py            # fairness, CIs, and metrics evaluation
├── models/                    # saved weights (.pkl)
├── results/                   # metrics.json, feature importances
└── notebooks/                 # original thesis notebooks
```

## 🚀 Exact Reproduction Commands

We use a modular Python pipeline that explicitly tracks fixed seeds, grid-search hyperparameter selection, and fixed test-splits.

To fully reproduce the pipeline from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run data preparation
python src/data_prep.py

# 3. Generate explicit test splits and scale features
python src/features.py

# 4. Train standard models + XGBoost (with 5-fold CV)
python src/train.py

# 5. Evaluate the models with uncertainty estimates
python src/evaluate.py
```

## 📈 1. The "Too Good To Be True" Baseline

Initially, the models were evaluated as-is. XGBoost was added as an honest improvement over the original models. The results showed near-perfect performance:

| Model | Accuracy | Accuracy (95% CI) | Precision | Sensitivity | AUC (95% CI) |
|-------|----------|-------------------|-----------|-------------|--------------|
| **XGBoost** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[1.000, 1.000]** |
| **Random Forest** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[1.000, 1.000]** |
| **SVC** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[0.999, 1.000]** |
| **Decision Tree**| **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[1.000, 1.000]** |
| **MLP** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[0.999, 1.000]** |
| **Logistic Reg.**| 0.984 | [0.953, 1.000] | 1.000 | 0.977 | [1.000, 1.000] |
| **KNN** | 0.800 | [0.692, 0.892] | 0.847 | 0.866 | [0.835, 0.974] |

## 🔍 2. Data Leakage Investigation

> [!WARNING]
> **Data Leakage Discovered:** Skeptical of perfect clinical results, a rigorous data audit was conducted. A shallow Decision Tree achieved 1.0 accuracy using a single split: `Duration > 0.0`. It was discovered that all Healthy Controls had a test duration of exactly `0.0`. See [DATA.md](docs/DATA.md) for a full breakdown. This confirmed the models were bypassing the complex biomechanical features and exploiting a data collection artifact.

## 🛠️ 3. The Correction & Honest Evaluation

To properly reproduce the clinical value of the spatial-temporal features (Velocity, Acceleration, AreaError, etc.), the leaked `Duration` column was explicitly dropped from the feature set. The pipeline was re-run, yielding the following true performance metrics:

| Model | Accuracy | Accuracy (95% CI) | Precision | Sensitivity | AUC (95% CI) |
|-------|----------|-------------------|-----------|-------------|--------------|
| **Random Forest** | **0.831** | [0.738, 0.908] | 0.840 | 0.933 | **[0.646, 0.916]** |
| **XGBoost**       | 0.754 | [0.646, 0.846] | 0.774 | 0.911 | [0.618, 0.893] |
| **Decision Tree** | 0.754 | [0.646, 0.862] | 0.854 | 0.778 | [0.625, 0.872] |
| **Logistic Reg.** | 0.692 | [0.569, 0.800] | 0.727 | 0.889 | [0.675, 0.903] |
| **SVC**           | 0.677 | [0.569, 0.785] | 0.731 | 0.844 | [0.515, 0.784] |
| **KNN**           | 0.662 | [0.538, 0.769] | 0.735 | 0.800 | [0.438, 0.736] |
| **MLP**           | 0.646 | [0.538, 0.754] | 0.739 | 0.756 | [0.580, 0.838] |

### 📊 Final Synthesis: Before & After Data Leak Correction

This table directly compares the artificial metrics generated by the `Duration` leak against the honest biomechanical baseline:

| Model | Accuracy (Leaked) | Accuracy (Honest) | Accuracy Diff | AUC (Leaked) | AUC (Honest) |
|-------|-------------------|-------------------|---------------|--------------|--------------|
| Random Forest | 1.000 | 0.831 | -0.169 | 1.000 | 0.788 |
| XGBoost | 1.000 | 0.754 | -0.246 | 1.000 | 0.762 |
| Decision Tree | 1.000 | 0.754 | -0.246 | 1.000 | 0.747 |
| Logistic Reg. | 0.984 | 0.692 | -0.292 | 1.000 | 0.798 |
| SVC | 1.000 | 0.677 | -0.323 | 0.999 | 0.658 |
| KNN | 0.800 | 0.662 | -0.138 | 0.900 | 0.588 |
| MLP | 1.000 | 0.646 | -0.354 | 0.999 | 0.716 |

This realistic baseline (~75% - 83%) validates the original 2023 thesis conclusions that differentiating healthy controls is challenging using purely biomechanical features, and demonstrates the necessity of rigorous Data QA in medical MLOps.

### Subgroup Fairness

Fairness was audited using the proxy feature `Dominant`. On the honest evaluation, Random Forest accuracy showed a noticeable performance gap across subgroups (0.889 vs 0.759). This indicates a potential statistical bias along handedness/laterality that requires further clinical investigation before deployment.
