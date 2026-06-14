import pandas as pd
import os

RAW_DATA_PATH = "data/raw/dataset.csv"
PROCESSED_DATA_PATH = "data/processed/dataset_cleaned.csv"

def prepare_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Expected raw data at {RAW_DATA_PATH}")
        
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Basic sanity checks: Drop rows where target PC is null
    df = df.dropna(subset=['PC'])
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Data prepped and saved to {PROCESSED_DATA_PATH} (Shape: {df.shape})")

if __name__ == "__main__":
    prepare_data()
