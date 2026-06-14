import pandas as pd
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from utils import set_seed

MODELS_DIR = "models/"

def train_models(seed=42):
    set_seed(seed)
    
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv", index_col=0)
    y_train = pd.read_csv("data/processed/y_train.csv", index_col=0).squeeze("columns")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Define models and their hyperparameter grids for tuning
    models = {
        "DecisionTree": (DecisionTreeClassifier(random_state=seed), {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }),
        "KNN": (KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=seed), {
            'C': [0.1, 1.0, 10.0]
        }),
        "SVC": (SVC(probability=True, random_state=seed), {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear']
        }),
        "MLP": (MLPClassifier(max_iter=1000, random_state=seed), {
            'hidden_layer_sizes': [(50,), (100,), (50,50)],
            'alpha': [0.0001, 0.001]
        }),
        "RandomForest": (RandomForestClassifier(random_state=seed), {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }),
        # Honest improvement: XGBoost
        "XGBoost": (XGBClassifier(eval_metric='logloss', random_state=seed), {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1]
        })
    }
    
    for name, (model, params) in models.items():
        print(f"Training {name}...")
        clf = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        print(f"  Best params: {clf.best_params_}")
        print(f"  Best CV AUC: {clf.best_score_:.4f}")
        
        # Save the best estimator
        joblib.dump(clf.best_estimator_, os.path.join(MODELS_DIR, f"{name}.pkl"))

if __name__ == "__main__":
    train_models()
