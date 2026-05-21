import os
import sys

if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Configuration ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = 'results'
REQUIRED_FIELDS = ("pclass", "sex", "age", "fare", "familySize", "embarked", "title")
EMBARKED_MAP = {'S': 2, 'C': 0, 'Q': 1}

@st.cache_resource
def get_available_models():
    """Returns a list of model directories and their metadata."""
    if not os.path.exists(MODEL_DIR):
        return []
    
    models = []
    for item in os.listdir(MODEL_DIR):
        item_path = os.path.join(MODEL_DIR, item)
        if os.path.isdir(item_path):
            meta_path = os.path.join(item_path, 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    models.append({
                        'dir': item_path,
                        'name': meta.get('model_name', item),
                        'accuracy': meta.get('accuracy', 0),
                        'metadata': meta
                    })
                except:
                    pass
    # Sort by accuracy descending
    models.sort(key=lambda x: x['accuracy'], reverse=True)
    return models

@st.cache_resource
def load_specific_model(model_dir):
    try:
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        scaler = None
        if metadata.get('uses_scaler'):
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            
        return model, metadata, scaler
    except Exception as e:
        st.error(f"Error loading model from {model_dir}: {e}")
        return None, None, None

# --- Helper Functions ---
def prepare_features(pclass, sex, age, fare, family_size, embarked, title, metadata):
    sex_code = 1 if sex == 'Male' else 0
    embarked_code = EMBARKED_MAP[embarked[0]] # Just take 'S', 'C', or 'Q'
    
    title_fallback = metadata['encodings']['title'].get('Mr', 1)
    title_code = metadata['encodings']['title'].get(title, title_fallback)
    
    input_df = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex_code,
        'Age': age,
        'Fare': fare,
        'FamilySize': family_size,
        'Embarked': embarked_code,
        'Title': title_code,
    }])

    # Derived features
    input_df['FarePerPerson_Log'] = np.log1p(input_df['Fare'] / input_df['FamilySize'])
    input_df['Age_Class'] = input_df['Age'] * input_df['Pclass']

    return input_df[metadata['features']]

# --- UI Layout ---

# Load image
img_path = os.path.join(os.path.dirname(__file__), 'assets', 'titanic_background.png')
if os.path.exists(img_path):
    st.image(img_path, use_container_width=True)

st.title("🚢 Titanic Survival Predictor")
st.markdown("""
Welcome to the Titanic Survival Prediction System. 
This dashboard uses a Machine Learning model trained on the historical dataset of Titanic passengers to predict your chances of survival based on your profile.
""")

available_models = get_available_models()

if not available_models:
    st.error("No models found! Please run the training script first.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.header("👤 Passenger Profile")
    
    with st.container():
        st.subheader("Personal Details")
        title = st.selectbox("Title", ['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Col', 'Major'])
        sex = st.radio("Gender", ['Male', 'Female'])
        age = st.slider("Age", min_value=0, max_value=100, value=30)
        family_size = st.number_input("Family Size (including yourself)", min_value=1, max_value=15, value=1)

    with st.container():
        st.subheader("Ticket Details")
        pclass_str = st.selectbox("Ticket Class", ['1st Class', '2nd Class', '3rd Class'], index=2)
        pclass = int(pclass_str[0])
        
        fare = st.number_input("Fare ($)", min_value=0.0, max_value=1000.0, value=32.0, step=5.0)
        
        embarked = st.selectbox("Port of Embarkation", [
            'Southampton (UK)', 
            'Cherbourg (France)', 
            'Queenstown (Ireland)'
        ])

    with st.container():
        st.subheader("Model Selection")
        model_options = {f"{m['name']} (Acc: {m['accuracy']*100:.1f}%)": m['dir'] for m in available_models}
        selected_model_label = st.selectbox("Choose Model", list(model_options.keys()))
        selected_model_dir = model_options[selected_model_label]
        
    # Load selected model dynamically
    model, metadata, scaler = load_specific_model(selected_model_dir)

    predict_btn = st.button("🔮 Predict Fate", type="primary", use_container_width=True)

with col2:
    st.header("📊 Prediction Results")
    
    if predict_btn:
        with st.spinner("Analyzing profile..."):
            try:
                # Prepare features
                X_pred = prepare_features(pclass, sex, age, fare, family_size, embarked, title, metadata)
                
                # Scale if needed
                if scaler is not None:
                    X_pred = scaler.transform(X_pred)
                    
                # Predict
                prediction = int(model.predict(X_pred)[0])
                
                # Calculate probability
                if hasattr(model, 'predict_proba'):
                    probability = float(model.predict_proba(X_pred)[0][1])
                elif hasattr(model, 'decision_function'):
                    dec = model.decision_function(X_pred)
                    probability = float(1 / (1 + np.exp(-dec[0])))
                else:
                    probability = 0.5
                    
                probability = float(np.clip(probability, 0.0, 1.0))
                
                # Display results
                st.markdown("---")
                if prediction == 1:
                    st.success("## 🎉 You Survived!")
                    st.balloons()
                else:
                    st.error("## 💀 You Perished.")
                
                # Display probability
                st.markdown(f"### Survival Probability: **{probability*100:.1f}%**")
                st.progress(probability)
                
                # Model Info
                with st.expander("Model Insights"):
                    st.write(f"**Model Used:** {metadata['model_name']}")
                    st.write(f"**Model Accuracy:** {metadata['accuracy']*100:.2f}%")
                    st.write("**Key Factors:**")
                    st.write("- Women and children generally had a much higher survival rate.")
                    st.write("- First-class passengers had priority access to lifeboats.")
                    st.write("- Larger families often struggled to stay together during the evacuation.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("👈 Enter your profile details on the left and click 'Predict Fate' to see if you would survive the Titanic.")
