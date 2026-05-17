# Titanic Survival Prediction

Small project demonstrating a Titanic survival prediction model with a minimal API and a Jupyter notebook.

## Summary

- Simple web UI and `prediction_api.py` for making predictions.
- Jupyter notebook: `Titanic Survival Prediction.ipynb` for exploration and training.

## Requirements

- Python 3.8 or newer
- See `requirements.txt` for Python dependencies

## Setup (create and use a virtual environment)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (Command Prompt):

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS / Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

- Start the API server:

```bash
python prediction_api.py
```

- Open the notebook in Jupyter to explore or re-train: `Titanic Survival Prediction.ipynb`

## Notes

- Use the created `.venv` for reproducible dependency management and to avoid polluting the system Python.
- If you prefer Conda, create a Conda env and install dependencies from `requirements.txt`.
