# 🚢 Titanic Survival Prediction

Welcome to the **Titanic Survival Prediction** project! This repository contains a machine learning model that predicts whether a passenger would have survived the Titanic disaster based on their details.

---

## 🌟 Features

- **🧠 Machine Learning Notebook**: A detailed Jupyter notebook (`titanic_survival_prediction_ml.ipynb`) for data exploration, preprocessing, and training various ML models (Logistic Regression, KNN, SVM, Decision Tree, Random Forest).
- **🎛️ Interactive CLI**: Use `main.py` to easily train new models from the command line and explore the results.
- **📊 Streamlit Dashboard**: A beautiful, interactive web UI built with Streamlit (`src/streamlit.py`) where you can enter passenger details and predict their survival chances instantly!
- **🤖 Automated Training Pipeline**: A GitHub Action (`.github/workflows/training.yml`) automatically triggers model training and saves the best models back to the `results/` folder upon any data or code updates.

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.8+
- Recommended: A virtual environment (venv or Conda)

### ⚙️ Installation

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/yourusername/titanic-survival-prediction-ml.git
   cd titanic-survival-prediction-ml
   \`\`\`

2. **Set up your virtual environment:**
   *(Windows PowerShell)*
   \`\`\`powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   \`\`\`

3. **Install dependencies:**
   \`\`\`bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   \`\`\`

---

## 🎮 Usage

You can manage everything from the main CLI menu!

Run the following command to open the interactive menu:
\`\`\`bash
python main.py
\`\`\`

**Menu Options:**
1. **Train Models**: Select and train a specific model (or all of them) on the Titanic dataset.
2. **Run Streamlit Dashboard**: Launches the interactive web app locally. 
3. **Exit**: Closes the application.

*To view the dashboard, simply choose option 2, and it will automatically open in your default web browser.*

---

## 📝 Notes
- Ensure your dataset is located at `data/titanic_passengers_data.csv` before training.
- Use the `.venv` to keep your system Python clean and for reproducible dependency management.
- Trained models and their metadata are saved in the `results/` folder and used by the Streamlit app.