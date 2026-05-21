import pandas as pd
import numpy as np
import os
import joblib
import json
import re

from src.models import train_models

# ---------------------------------------------------------
# DATA PROCESSING (from data.py)
# ---------------------------------------------------------
def load_data(file_path='data/titanic_passengers_data.csv'):
    """Load the Titanic dataset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    return pd.read_csv(file_path)

def engineer_features(df):
    """Perform feature engineering on the Titanic dataset."""
    df_clean = df.copy()

    # Handle missing values
    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())

    if 'Fare' in df_clean.columns and 'FamilySize' in df_clean.columns:
        df_clean['FarePerPerson'] = df_clean['Fare'] / df_clean['FamilySize']
        df_clean['FarePerPerson_Log'] = np.log1p(df_clean['FarePerPerson'])
        df_clean['FarePerPerson_Log'] = df_clean['FarePerPerson_Log'].fillna(df_clean['FarePerPerson_Log'].median())

    if 'Age' in df_clean.columns and 'Pclass' in df_clean.columns:
        df_clean['Age_Class'] = df_clean['Age'] * df_clean['Pclass']
        df_clean['Age_Class'] = df_clean['Age_Class'].fillna(df_clean['Age_Class'].median())

    # Drop redundant columns
    cols_to_drop = ['SibSp', 'Parch', 'Name', 'PassengerId', 'Cabin']
    df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')

    # Encode categorical variables
    if 'Sex' in df_clean.columns:
        df_clean['Sex'] = df_clean['Sex'].map({'male': 1, 'female': 0})

    if 'Title' in df_clean.columns:
        title_mapping = {
            'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Dr': 5,
            'Rev': 6, 'Col': 7, 'Major': 8, 'Mlle': 3, 'Countess': 2,
            'Ms': 2, 'Don': 7, 'Dona': 2, 'Mme': 2, 'Capt': 7, 'Jonkheer': 7
        }
        df_clean['Title'] = df_clean['Title'].map(title_mapping).fillna(1) # Default to Mr

    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
        embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
        df_clean['Embarked'] = df_clean['Embarked'].map(embarked_mapping)

    if 'AgeGroup' in df_clean.columns:
        agegroup_mapping = {'Child': 1, 'Teenager': 2, 'Young Adult': 3, 'Adult': 4, 'Middle-aged': 5, 'Senior': 6}
        df_clean['AgeGroup'] = df_clean['AgeGroup'].map(agegroup_mapping)

    if 'FareGroup' in df_clean.columns:
        faregroup_mapping = {'Low': 1, 'High': 2, 'Very High': 3}
        df_clean['FareGroup'] = df_clean['FareGroup'].map(faregroup_mapping)

    if 'Deck' in df_clean.columns:
        deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
        df_clean['Deck'] = df_clean['Deck'].map(deck_mapping)

    # Select final features
    fare_col = 'FarePerPerson_Log' if 'FarePerPerson_Log' in df_clean.columns else 'FarePerPerson'
    feature_cols = ['Pclass', 'Sex', 'Age', fare_col, 'FamilySize', 'Embarked', 'Title', 'Age_Class']

    # Ensure all required features are present
    missing_cols = [col for col in feature_cols if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for modeling: {missing_cols}")

    X = df_clean[feature_cols].copy()
    y = df_clean['Survived'].copy() if 'Survived' in df_clean.columns else None

    if y is not None:
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

    # Handle remaining NaN values
    X = X.dropna()
    if y is not None:
        y = y.loc[X.index]

    return X, y, feature_cols

# ---------------------------------------------------------
# UTILITIES (from utils.py)
# ---------------------------------------------------------
def save_model(model, model_name, scaler, accuracy, f1_score, feature_cols, output_dir='results'):
    """Save the model, scaler, and metadata to the output directory."""
    safe_name = re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')
    model_dir = os.path.join(output_dir, safe_name)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model
    model_path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f" Model {model_name} saved to {model_path}")

    # Save the scaler if needed
    uses_scaler = model_name not in ["Decision Tree", "Random Forest"]
    if uses_scaler:
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)

    # Save metadata
    encoding_info = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'f1_score': float(f1_score),
        'features': feature_cols,
        'uses_scaler': uses_scaler,
        'encodings': {
            'sex': {'male': 1, 'female': 0},
            'embarked': {'C': 0, 'Q': 1, 'S': 2},
            'title': {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 8},
            'deck': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'U': 0}
        }
    }

    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(encoding_info, f, indent=2)
    print(f" Metadata for {model_name} saved to {metadata_path}")

# ---------------------------------------------------------
# MAIN ENTRY POINT (from train.py)
# ---------------------------------------------------------
def generate_and_export_model(data_path='data/titanic_passengers_data.csv', output_dir='results', model_choice='all'):
    """Generate and export the machine learning models."""
    print("=" * 50)
    print("MODEL GENERATION AND EXPORT")
    print("=" * 50)
    
    print("Loading data...")
    df = load_data(data_path)

    print("Engineering features...")
    X, y, feature_cols = engineer_features(df)

    print("Training and evaluating models...")
    results, scaler = train_models(X, y, model_choice)

    if results:
        best = results[0]
        print(f"\n Best Model Evaluated: {best['Model']} (Accuracy: {best['Accuracy']*100:.2f}%)")

        for i, res in enumerate(results):
            save_model(res['Model_Object'], res['Model'], scaler, res['Accuracy'], res['F1-Score'], feature_cols, output_dir)
    else:
        print("\n No models were trained.")
