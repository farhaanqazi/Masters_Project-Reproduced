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

## 📈 Results Overview (with Uncertainty)

> [!WARNING]
> **Data Leakage Discovered:** The perfect 1.000 accuracy scores shown below are the result of a severe data collection artifact in the original dataset's `Duration` column (all Healthy Controls have a duration of exactly 0.0). See [DATA.md](docs/DATA.md) for a full breakdown. The immediate next step for this project is to drop the `Duration` feature and re-evaluate the models on the true biomechanical signals.

The models were evaluated based on multiple metrics, with 95% Confidence Intervals calculated via bootstrapping (N=1000). XGBoost was added as an honest improvement over the original models.

| Model | Accuracy | Accuracy (95% CI) | Precision | Sensitivity | AUC (95% CI) |
|-------|----------|-------------------|-----------|-------------|--------------|
| **XGBoost** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[1.000, 1.000]** |
| **Random Forest** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[1.000, 1.000]** |
| **SVC** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[0.999, 1.000]** |
| **Decision Tree**| **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[1.000, 1.000]** |
| **MLP** | **1.000** | [1.000, 1.000] | 1.000 | 1.000 | **[0.999, 1.000]** |
| **Logistic Reg.**| 0.984 | [0.953, 1.000] | 1.000 | 0.977 | [1.000, 1.000] |
| **KNN** | 0.800 | [0.692, 0.892] | 0.847 | 0.866 | [0.835, 0.974] |

### Subgroup Fairness

Fairness was audited using the proxy feature `Dominant`. For the top models (XGBoost, RandomForest, SVC, DecisionTree, MLP), accuracy was **1.0 across both subgroups**, showing no observed statistical bias along this axis on the testing set.
