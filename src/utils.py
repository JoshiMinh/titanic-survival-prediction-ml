import os
import joblib
import json
import re

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
