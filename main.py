import sys
import os
from src.pipeline import generate_and_export_model

# --- CLI Menu ---
def print_menu():
    print("\n" + "="*50)
    print(" TITANIC SURVIVAL PREDICTION - MAIN MENU")
    print("="*50)
    print("1. Train Models (single option or all)")
    print("2. Run Streamlit Dashboard")
    print("3. Exit")
    print("="*50)

def main():
    while True:
        print_menu()
        choice = input("Enter your choice (1-3): ").strip()

        if choice == '1':
            data_file = input("Enter path to dataset [default: data/titanic_passengers_data.csv]: ").strip()
            if not data_file:
                data_file = 'data/titanic_passengers_data.csv'
            
            print("\nWhich model would you like to train?")
            print("1. All Models")
            print("2. Logistic Regression")
            print("3. KNN")
            print("4. SVM")
            print("5. Decision Tree")
            print("6. Random Forest")
            model_choice = input("Enter your choice (1-6) [default: 1]: ").strip()
            
            choice_map = {
                '1': 'all',
                '2': 'Logistic Regression',
                '3': 'KNN',
                '4': 'SVM',
                '5': 'Decision Tree',
                '6': 'Random Forest'
            }
            selected_model = choice_map.get(model_choice, 'all')
            
            try:
                generate_and_export_model(data_path=data_file, output_dir='results', model_choice=selected_model)
            except Exception as e:
                print(f"\n Error during training: {e}")
                
        elif choice == '2':
            print("Starting Streamlit Dashboard...")
            os.system("streamlit run src/streamlit.py")
        elif choice == '3':
            print("Exiting...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()
