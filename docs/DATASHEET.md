# Datasheet for Parkinson's Disease Biomechanical Dataset

## Motivation
- **For what purpose was the dataset created?** To evaluate machine learning models in diagnosing Parkinson's Disease from biomechanical and motor-skill tests.
- **Who created the dataset?** Inherited from a prior Master's thesis / GitHub repository (`farhaanqazi/F21MP`).
- **Who funded the creation of the dataset?** Not explicitly stated.

## Composition
- **What do the instances that comprise the dataset represent?** Individual clinical motor-skill trials.
- **How many instances are there?** 325 rows (trials).
- **Does the dataset contain all possible instances?** No, it represents a specific sampled cohort.
- **What data does each instance consist of?** ID, Dominant (Handedness proxy), Attempts, PC (Parkinson's Control target), Duration, Time, AreaError, TimeTriangles 1-5, Distance, LeaveSurface, Side, TimeContact, ZeroVel, ZeroAcc.
- **Is any information missing from individual instances?** No overt missing values in key columns.

## Collection Process
- **How was the data associated with each instance acquired?** Unknown specific hardware, likely a digital pen or digitizing tablet capturing spatial-temporal parameters.
- **Over what timeframe was the data collected?** Unknown.

## Preprocessing/Cleaning/Labeling
- **Was any preprocessing/cleaning/labeling of the data done?** The raw `dataset.csv` already contains aggregated spatial-temporal features rather than raw time-series data. Target `PC` is pre-assigned.

## Uses
- **Has the dataset been used for any tasks already?** Yes, it was used to train baseline Decision Tree, SVM, KNN, and Random Forest models in the original repository.
- **What tasks could the dataset be used for?** Diagnostic classification, progression tracking (if longitudinal, though this seems cross-sectional), or motor-skill characterization.

## Distribution & Maintenance
- **Will the dataset be distributed to third parties?** Available publicly via GitHub.
- **Who maintains the dataset?** The repository owner.
