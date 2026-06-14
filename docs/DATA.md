# Data Overview

## Dataset Source
The data was originally provided as a `.csv` file in the original Master's project repository:
`https://raw.githubusercontent.com/farhaanqazi/F21MP/d2fc9961c91ef0f0472dc7afef8e0b52e7d557e4/PDdataset.csv`

## Access Procedure
The dataset is openly accessible via the repository. It has been moved to the local `data/raw/dataset.csv` path within this reproduction repository.

## Dataset Description
This dataset contains biomechanical and movement-based features captured during motor-skill tests, which are used to classify individuals as having Parkinson's Disease (`PC=1`) or Healthy Controls (`PC=0`).

### Features:
- `ID`: Subject ID
- `Dominant`: Dominant hand indicator
- `Attempts`: Number of attempts
- `PC`: **Target variable** (Parkinson's Disease / Control)
- `Duration`, `Time`, `AreaError`, `TimeTriangles_1` to `TimeTriangles_5`, `Distance`, `LeaveSurface`, `Side`, `TimeContact`, `ZeroVel`, `ZeroAcc`: Biomechanical feature metrics.

## Exclusions
No rows were explicitly dropped during data loading. The data is processed as-is, adhering to the original thesis pipeline.

## ⚠️ Known Data Leakage Artifact

**Critical Finding:** The `Duration` feature contains a massive data collection artifact that acts as a perfect predictor for the target variable (`PC`).
* For every **Healthy Control** (`PC=0`), `Duration` is exactly `0.0`.
* For every **Parkinson's Disease patient** (`PC=1`), `Duration` is strictly `> 0.0`.

This implies that the test duration was likely not recorded for the control group during the original study. Consequently, machine learning models using this feature will achieve exactly 1.0 accuracy by simply thresholding `Duration > 0.0`, completely bypassing the actual biomechanical movement metrics. 

**Mitigation / Next Steps:** To evaluate the true predictive power of the motor-skill features, the `Duration` column must be explicitly dropped from the feature set prior to training.
