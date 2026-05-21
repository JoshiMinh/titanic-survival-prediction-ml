import os
import joblib
import json

def save_model(best_model, best_model_name, scaler, accuracy, f1_score, feature_cols, output_dir='results'):
    """Save the model, scaler, and metadata to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the model
    model_path = os.path.join(output_dir, 'titanic_model.pkl')
    joblib.dump(best_model, model_path)
    print(f" Model saved to {model_path}")

    # Save the scaler if needed
    uses_scaler = best_model_name not in ["Decision Tree", "Random Forest"]
    if uses_scaler:
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f" Scaler saved to {scaler_path}")
    else:
        print(" No scaler needed for tree-based models")

    # Save metadata
    encoding_info = {
        'model_name': best_model_name,
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

    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(encoding_info, f, indent=2)
    print(f" Metadata saved to {metadata_path}")
    print("\n Model export complete!")
