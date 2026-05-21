from src.data import load_data, engineer_features
from src.models import train_models
from src.utils import save_model

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
