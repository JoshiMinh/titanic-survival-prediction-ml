import sys
import os
import joblib
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.model_generator import generate_and_export_model

# --- Configuration ---
MODEL_DIR = 'results'
REQUIRED_FIELDS = ("pclass", "sex", "age", "fare", "familySize", "embarked", "title")
EMBARKED_MAP = {'S': 2, 'C': 0, 'Q': 1}

# --- Global Variables for API ---
app = Flask(__name__)
CORS(app)
model = None
metadata = None
scaler = None
TITLE_FALLBACK = 1

def load_model_and_metadata():
    """Load model, metadata and scaler from the results directory."""
    global model, metadata, scaler, TITLE_FALLBACK
    
    if not os.path.exists(os.path.join(MODEL_DIR, 'titanic_model.pkl')):
        print(f" Model not found in '{MODEL_DIR}'. Please train the model first.")
        return False
        
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'titanic_model.pkl'))
        with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)

        TITLE_FALLBACK = metadata['encodings']['title'].get('Mr', 1)
        
        if metadata.get('uses_scaler'):
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        else:
            scaler = None
            
        return True
    except Exception as e:
        print(f" Error loading model: {e}")
        return False

# --- Routes ---
@app.route('/api')
def api_home():
    """API information endpoint."""
    if not metadata:
        return jsonify({'status': 'offline', 'error': 'Model not loaded'})
        
    return jsonify({
        'status': 'online',
        'model': metadata['model_name'],
        'accuracy': metadata['accuracy'],
        'endpoints': ['/api/predict', '/api/model-info']
    })

@app.route('/api/model-info')
def model_info():
    """Return model metadata."""
    if not metadata:
        return jsonify({'error': 'Model not loaded'}), 500
    return jsonify(metadata)

# --- Feature Preparation ---
def _prepare_features(data: dict) -> pd.DataFrame:
    """Validate and convert incoming request JSON into model-ready features."""
    if not isinstance(data, dict):
        raise ValueError('Request body must be a JSON object.')

    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    try:
        pclass = int(data['pclass'])
        sex = str(data['sex']).lower()
        if sex not in ('male', 'female'):
            raise ValueError("sex must be 'male' or 'female'")
        sex_code = 1 if sex == 'male' else 0

        age = float(data['age'])
        fare = float(data['fare'])
        family_size = int(data['familySize'])
        if family_size <= 0:
            raise ValueError('familySize must be greater than 0')

        embarked = str(data['embarked']).upper()
        if embarked not in EMBARKED_MAP:
            raise ValueError("embarked must be one of C, Q, S")
        embarked_code = EMBARKED_MAP[embarked]

        title_code = metadata['encodings']['title'].get(str(data['title']), TITLE_FALLBACK)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid field values: {exc}") from exc

    input_df = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex_code,
        'Age': age,
        'Fare': fare,
        'FamilySize': family_size,
        'Embarked': embarked_code,
        'Title': title_code,
    }])

    input_df['FarePerPerson_Log'] = np.log1p(input_df['Fare'] / input_df['FamilySize'])
    input_df['Age_Class'] = input_df['Age'] * input_df['Pclass']

    return input_df[metadata['features']]

# --- Prediction Endpoint ---
@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    data = request.get_json(silent=True)
    try:
        X_pred = _prepare_features(data)
        if scaler is not None:
            X_pred = scaler.transform(X_pred)

        prediction = int(model.predict(X_pred)[0])

        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(X_pred)[0][1])
        elif hasattr(model, 'decision_function'):
            dec = model.decision_function(X_pred)
            probability = float(1 / (1 + np.exp(-dec[0])))
        else:
            probability = 0.5

        probability = float(np.clip(probability, 0.0, 1.0))
        confidence = 'High' if abs(probability - 0.5) > 0.2 else 'Moderate'

        return jsonify({
            'survived': bool(prediction),
            'probability': probability,
            'confidence': confidence,
            'model': metadata['model_name']
        })

    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        return jsonify({'error': f'Unexpected error: {exc}'}), 500

# --- CLI Menu ---
def print_menu():
    print("\n" + "="*50)
    print(" TITANIC SURVIVAL PREDICTION - MAIN MENU")
    print("="*50)
    print("1. Train Model & Export")
    print("2. Run API Server")
    print("3. Run Streamlit Dashboard")
    print("4. Exit")
    print("="*50)

def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            data_file = input("Enter path to dataset [default: data/titanic_detailed_passengers_data.csv]: ").strip()
            if not data_file:
                data_file = 'data/titanic_detailed_passengers_data.csv'
            
            try:
                generate_and_export_model(data_path=data_file, output_dir=MODEL_DIR)
            except Exception as e:
                print(f"\n Error during training: {e}")
                
        elif choice == '2':
            if load_model_and_metadata():
                print("\n" + "="*50)
                print(" TITANIC SURVIVAL PREDICTION API SERVER")
                print("="*50)
                print(f" Model loaded: {metadata['model_name']}")
                print(f" Model accuracy: {metadata['accuracy']*100:.2f}%")
                print()
                print(" Server running at: http://localhost:5000")
                print(" API endpoint: http://localhost:5000/api/predict")
                print()
                print("Press Ctrl+C to stop")
                print("="*50 + "\n")
                
                try:
                    app.run(debug=False, port=5000, host='0.0.0.0')
                except KeyboardInterrupt:
                    print("\nServer stopped.")
            
        elif choice == '3':
            print("Starting Streamlit Dashboard...")
            os.system("streamlit run src/dashboard.py")
        elif choice == '4':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == '__main__':
    main()
