# Model Card: Parkinson's Disease Classification

## Model Details
- **Model Types**: Random Forest, Support Vector Classifier (SVC), Logistic Regression, K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), Decision Tree.
- **Model Version**: 1.0 (Refactored for reproducibility)
- **Developed by**: Base models from original thesis (`farhaanqazi/F21MP`); refactored for reproducibility.
- **Task**: Binary Classification (1 = Parkinson's Disease, 0 = Healthy Control).

## Intended Use
- **Primary Use**: Diagnostic screening classification based on extracted motor and biomechanical features (tablet/pen based).
- **Primary Users**: Medical researchers, data scientists testing baseline algorithms.
- **Out of Scope Uses**: Not to be used as a standalone clinical diagnostic tool without extensive prospective, multi-center clinical trials and regulatory approval.

## Factors
- **Demographics**: Unknown demographics in dataset (age, sex, ethnicity are not provided, presenting a significant limitation for fairness evaluation).
- **Subgroups**: We evaluate performance variations across `Side` and `Dominant` proxy features to establish baseline fairness.

## Metrics
- **Model Performance Measures**: Area Under the ROC Curve (AUC), Precision, Sensitivity (Recall/TPR), Specificity (TNR), Accuracy (Train/Test/Cross-Val).
- **Decision Threshold**: Standard 0.5 threshold for probability-based models.

## Evaluation Data
- **Dataset**: `dataset.csv`
- **Split**: 80% train, 20% test, generated with a fixed random seed.
- **Cross-Validation**: 5-fold CV used during hyperparameter tuning on the training set.

## Training Data
- Same source as evaluation data. No external data used.

## Quantitative Analyses
- *Results to be populated post-training in the main README.*

## Ethical Considerations
- Lack of demographic data prevents rigorous intersectional bias audits. The model could harbor hidden biases against certain age or ethnic groups.
- The diagnosis of Parkinson's Disease is sensitive; false positives can cause distress, while false negatives delay treatment. High sensitivity is prioritized for screening.
