import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from utils import set_seed

MODELS_DIR = "models/"
RESULTS_DIR = "results/"

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstraps=1000, seed=42):
    """Calculates 95% Confidence Interval for a metric using bootstrapping."""
    np.random.seed(seed)
    scores = []
    n_samples = len(y_true)
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    
    for _ in range(n_bootstraps):
        indices = np.random.randint(0, n_samples, n_samples)
        if len(np.unique(y_true_np[indices])) < 2:
            continue
        score = metric_func(y_true_np[indices], y_pred_np[indices])
        scores.append(score)
        
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return lower, upper

def evaluate_models(seed=42):
    set_seed(seed)
    
    X_test = pd.read_csv("data/processed/X_test.csv", index_col=0)
    y_test = pd.read_csv("data/processed/y_test.csv", index_col=0).squeeze("columns")
    
    has_dominant = 'Dominant' in X_test.columns
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {}
    
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    for mf in model_files:
        name = mf.replace(".pkl", "")
        model = joblib.load(os.path.join(MODELS_DIR, mf))
        
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        acc_ci = bootstrap_metric(y_test, y_pred, accuracy_score)
        auc_ci = bootstrap_metric(y_test, y_proba, roc_auc_score)
        
        results[name] = {
            "Accuracy": acc,
            "Accuracy_95CI": [acc_ci[0], acc_ci[1]],
            "Precision": prec,
            "Sensitivity (Recall)": rec,
            "AUC": auc,
            "AUC_95CI": [auc_ci[0], auc_ci[1]]
        }
        
        # Fairness/Subgroup analysis
        if has_dominant:
            subgroup_0_mask = X_test['Dominant'] < 0
            subgroup_1_mask = X_test['Dominant'] >= 0
            
            if sum(subgroup_0_mask) > 0 and sum(subgroup_1_mask) > 0:
                acc_0 = accuracy_score(y_test[subgroup_0_mask], y_pred[subgroup_0_mask])
                acc_1 = accuracy_score(y_test[subgroup_1_mask], y_pred[subgroup_1_mask])
                results[name]["Fairness_Dominant_0_Acc"] = acc_0
                results[name]["Fairness_Dominant_1_Acc"] = acc_1

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {RESULTS_DIR}metrics.json")
    
    # 1. Save Comparison.csv
    df_results = pd.DataFrame.from_dict(results, orient='index')
    cols = ['Accuracy', 'Precision', 'Sensitivity (Recall)', 'AUC']
    df_csv = df_results[cols].copy()
    df_csv.index.name = 'Model'
    df_csv.to_csv(os.path.join(RESULTS_DIR, "Comparison.csv"))
    
    # 2. Generate Comparison Charts
    plt.figure(figsize=(12, 8))
    
    models = list(df_csv.index)
    metrics_to_plot = cols
    
    n_models = len(models)
    n_metrics = len(metrics_to_plot)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.15
    index = np.arange(n_models)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, metric in enumerate(metrics_to_plot):
        values = df_csv[metric].values
        ax.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i])
        
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title("Comparison of ML Models for Parkinson's Diagnosis", fontsize=14)
    ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "Comparison_Charts.png"))
    plt.close('all')
    
    for model_name in ["RandomForest", "XGBoost"]:
        if model_name in results:
            model = joblib.load(os.path.join(MODELS_DIR, f"{model_name}.pkl"))
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 6))
                plt.title(f"Feature Importances ({model_name})")
                plt.bar(range(X_test.shape[1]), importances[indices], align="center")
                plt.xticks(range(X_test.shape[1]), X_test.columns[indices], rotation=90)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_feature_importance.png"))
                plt.close()

if __name__ == "__main__":
    evaluate_models()
