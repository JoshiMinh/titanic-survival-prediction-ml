from src.data import load_data, engineer_features
from src.models import train_models
from src.utils import save_model

def generate_and_export_model(data_path='data/titanic_passengers_data.csv', output_dir='results'):
    """Generate and export the best machine learning model."""
    print("=" * 50)
    print("MODEL GENERATION AND EXPORT")
    print("=" * 50)
    
    print("Loading data...")
    df = load_data(data_path)

    print("Engineering features...")
    X, y, feature_cols = engineer_features(df)

    print("Training and evaluating models...")
    best_model, best_model_name, scaler, accuracy, f1_score = train_models(X, y)

    print(f"\n Best Model: {best_model_name} (Accuracy: {accuracy*100:.2f}%)")

    save_model(best_model, best_model_name, scaler, accuracy, f1_score, feature_cols, output_dir)
