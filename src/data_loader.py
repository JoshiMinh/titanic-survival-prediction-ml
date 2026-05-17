import pandas as pd
import os

def load_data(file_path='titanic_detailed_passengers_data.csv'):
    """Load the Titanic dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)
