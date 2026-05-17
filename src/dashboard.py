import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

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

# --- Load Model & Metadata ---
@st.cache_resource
def load_model_data():
    if not os.path.exists(os.path.join(MODEL_DIR, 'titanic_model.pkl')):
        return None, None, None
        
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'titanic_model.pkl'))
        with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
            metadata = json.load(f)
            
        scaler = None
        if metadata.get('uses_scaler'):
            scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
            
        return model, metadata, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, metadata, scaler = load_model_data()

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

if not model:
    st.error("Model not found! Please run the training script first.")
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
