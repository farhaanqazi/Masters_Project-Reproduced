import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import set_seed

PROCESSED_DATA_PATH = "data/processed/dataset_cleaned.csv"
SPLITS_DIR = "splits/"

def process_features_and_split(seed=42):
    set_seed(seed)
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Run data_prep.py first. Missing {PROCESSED_DATA_PATH}")
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Target column is 'PC'
    X = df.drop(columns=['ID', 'PC'])
    y = df['PC']
    
    # Robust 80/20 train/test split, stratified by target to handle imbalance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    # Save the splits indices explicitly to splits/
    os.makedirs(SPLITS_DIR, exist_ok=True)
    splits = {
        "train_indices": list(X_train.index),
        "test_indices": list(X_test.index),
        "seed": seed
    }
    with open(os.path.join(SPLITS_DIR, "split_indices.json"), "w") as f:
        json.dump(splits, f, indent=4)
        
    # Standard Scaling
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    # Save processed features to data/processed
    X_train_scaled.to_csv("data/processed/X_train.csv")
    X_test_scaled.to_csv("data/processed/X_test.csv")
    y_train.to_csv("data/processed/y_train.csv")
    y_test.to_csv("data/processed/y_test.csv")
    
    print(f"Generated train/test splits. Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Indices saved to {SPLITS_DIR}split_indices.json")
    print(f"Scaled features saved to data/processed/")

if __name__ == "__main__":
    process_features_and_split()
