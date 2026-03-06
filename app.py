import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cystectomy Risk Calculator", page_icon="🩺", layout="wide")

# --- HEADER & DISCLAIMER ---
st.title("Cystectomy 30-Day Risk Calculator")
st.markdown("""
**Predictive modeling for 30-day Mortality and Major Morbidity following Radical and Partial Cystectomy.**
""")

st.warning("""
**⚠️ FOR RESEARCH PURPOSES ONLY** This application is a demonstration of machine learning models (Random Forest and XGBoost) trained on ACS-NSQIP data. 
It is not intended to diagnose, treat, cure, or prevent any disease, nor should it replace the clinical judgment of a licensed healthcare professional.
""")

# --- LOAD MODELS ---
# We use st.cache_resource so it only loads the models once
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        rf_mort = joblib.load('mortality_rf_model.pkl')
        xgb_morb = joblib.load('morbidity_xgb_model.pkl')
        return preprocessor, rf_mort, xgb_morb
    except FileNotFoundError:
        return None, None, None

preprocessor, rf_mort, xgb_morb = load_models()

# --- USER INTERFACE (CLINICAL INPUTS) ---
st.header("Patient Characteristics & Surgical Details")

# Row 1: Demographics & Habitus
st.subheader("Demographics & Physical Status")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age (Years)", min_value=18, max_value=100, value=65)
    sex = st.radio("Biological Sex", options=["Male", "Female"])
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0)

with col2:
    asa = st.selectbox("ASA Physical Status Classification", options=[
        "1 - A normal healthy patient",
        "2 - A patient with mild systemic disease",
        "3 - A patient with severe systemic disease",
        "4 - A patient with severe systemic disease that is a constant threat to life",
        "5 - A moribund patient who is not expected to survive without the operation"
    ], index=2)
    
    fnstatus = st.selectbox("Pre-operative Functional Status", options=[
        "Independent", "Partially Dependent", "Totally Dependent"
    ])

with col3:
    race = st.selectbox("Race", options=["White", "Black or African American", "Asian", "American Indian", "Other/Unknown"])
    hispanic = st.radio("Hispanic Ethnicity?", options=["No", "Yes"])

# Row 2: Surgical Details
st.subheader("Surgical Factors")
cpt_code = st.selectbox("Primary CPT Code (Surgical Procedure)", options=[
    "51550 - Partial cystectomy",
    "51590 - Radical cystectomy; with ureteroileal conduit",
    "51595 - Radical cystectomy; with continent diversion",
    "51596 - Radical cystectomy; with continent diversion with Neobladder"
])

col_s1, col_s2 = st.columns(2)
with col_s1:
    neoadj = st.radio("Neoadjuvant Chemotherapy prior to surgery?", options=["No", "Yes"])
    prior_pelvic = st.radio("Prior Pelvic Surgery?", options=["No", "Yes"])
with col_s2:
    prior_radio = st.radio("Prior Pelvic Radiation?", options=["No", "Yes"])

# Row 3: Pre-operative Labs
st.subheader("Pre-operative Laboratory Values")
lab1, lab2, lab3, lab4 = st.columns(4)
with lab1:
    pralbum = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=4.0, step=0.1)
with lab2:
    prcreat = st.number_input("Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=1.0, step=0.1)
with lab3:
    prhct = st.number_input("Hematocrit (%)", min_value=15.0, max_value=60.0, value=40.0, step=1.0)
with lab4:
    prplate = st.number_input("Platelets (x10^9/L)", min_value=10, max_value=999, value=250, step=10)

# Row 4: Comorbidities
st.subheader("Medical Comorbidities")
c1, c2, c3 = st.columns(3)
with c1:
    hypermed = st.radio("Hypertension Requiring Medication?", options=["No", "Yes"])
    smoke = st.radio("Current Smoker (within 1 year)?", options=["No", "Yes"])
    dm = st.selectbox("Diabetes Mellitus", options=["No", "Yes - Non-Insulin", "Yes - Insulin"])
with c2:
    hxcopd = st.radio("Severe COPD?", options=["No", "Yes"])
    hxchf = st.radio("Congestive Heart Failure (within 30 days)?", options=["No", "Yes"])
    renafail = st.radio("Acute Renal Failure?", options=["No", "Yes"])
with c3:
    dialysis = st.radio("Currently on Dialysis?", options=["No", "Yes"])
    ascites = st.radio("Ascites present?", options=["No", "Yes"])
    wtloss = st.radio("More than 10% weight loss in last 6 months?", options=["No", "Yes"])

# --- DATA PROCESSING & PREDICTION ---
st.markdown("---")
if st.button("Calculate 30-Day Risk", type="primary", use_container_width=True):
    if preprocessor is None:
        st.error("Model files not found! Please ensure 'preprocessor.pkl', 'mortality_rf_model.pkl', and 'morbidity_xgb_model.pkl' are in the same directory.")
    else:
        # We need the exact list of predictors used in training
        predictors = ['AGE', 'SEX', 'Race', 'Ethnicity_Hispanic', 'ASACLAS', 'FNSTATUS2', 'BMI', 
                      'DM', 'SMOKE', 'HXCOPD', 'HXCHF', 'HYPERMED', 'RENAFAIL', 'DIALYSIS', 
                      'ASCITES', 'WTLOSS', 'PRALBUM', 'PRCREAT', 'PRHCT', 'PRPLATE', 
                      'Neoadj_chemo', 'CYST_PRIOR_PRADIO', 'PRIORpelvicsurgery', 'CPT']

        # CRITICAL: These inputs must exactly match the format found in your Excel file.
        # If your Excel file used "Yes" and "No", pass the string directly without changing it to 1 or 0.
        # If your Excel used 1 and 0, keep your original list comprehension. 
        # (Assuming standard NSQIP formatting below)
        # Map UI text inputs back to the exact numerical/categorical format the model expects
        input_dict = {
            'AGE': [float(age)],
            'SEX': [0 if sex == "Female" else 1],  
            'Race': [race],  # Assumes your Excel used strings like "White"
            'Ethnicity_Hispanic': [1 if hispanic == "Yes" else 0],
            'ASACLAS': [float(asa.split(" -")[0])],
            'FNSTATUS2': [fnstatus], 
            'BMI': [float(bmi)],
            'DM': [0 if dm == "No" else (1 if "Non-Insulin" in dm else 2)],
            'SMOKE': [1 if smoke == "Yes" else 0],
            'HXCOPD': [1 if hxcopd == "Yes" else 0],
            'HXCHF': [1 if hxchf == "Yes" else 0],
            'HYPERMED': [1 if hypermed == "Yes" else 0],
            'RENAFAIL': [1 if renafail == "Yes" else 0],
            'DIALYSIS': [1 if dialysis == "Yes" else 0],
            'ASCITES': [1 if ascites == "Yes" else 0],
            'WTLOSS': [1 if wtloss == "Yes" else 0],
            'PRALBUM': [float(pralbum)],
            'PRCREAT': [float(prcreat)],
            'PRHCT': [float(prhct)],
            'PRPLATE': [float(prplate)],
            'Neoadj_chemo': [1 if neoadj == "Yes" else 0],
            'CYST_PRIOR_PRADIO': [1 if prior_radio == "Yes" else 0],
            'PRIORpelvicsurgery': [1 if prior_pelvic == "Yes" else 0],
            'CPT': [str(cpt_code.split(" -")[0])]  # CRITICAL: Must be a string!
        }
        
        input_df = pd.DataFrame(input_dict)
        
        try:
            # 1. Transform the raw inputs through your saved preprocessor
            X_processed_array = preprocessor.transform(input_df)
            
            # 2. Reconstruct the exact feature names the model expects
            numeric_features = ['AGE', 'BMI', 'PRALBUM', 'PRCREAT', 'PRHCT', 'PRPLATE']
            categorical_features = [c for c in predictors if c not in numeric_features]
            ohe_names = list(preprocessor.named_transformers_['cat']['ohe'].get_feature_names_out(categorical_features))
            feature_names = numeric_features + ohe_names
            
            # 3. Convert back to a DataFrame
            final_input_df = pd.DataFrame(X_processed_array, columns=feature_names)
            
            # 4. Generate Predictions
            mort_risk = rf_mort.predict_proba(final_input_df)[0][1]
            morb_risk = xgb_morb.predict_proba(final_input_df)[0][1]
            
            st.success("Analysis Complete!")
            
            res1, res2 = st.columns(2)
            with res1:
                st.metric("Estimated 30-Day Mortality Risk", f"{mort_risk * 100:.1f}%")
                if mort_risk >= 0.05:
                    st.error("⚠️ High risk of post-operative mortality. Consider ICU-level care post-operatively or re-evaluation of surgical candidacy.")
                elif mort_risk >= 0.02:
                    st.warning("Elevated risk of mortality. Standard post-operative precautions advised.")
            
            with res2:
                st.metric("Estimated 30-Day Major Morbidity Risk", f"{morb_risk * 100:.1f}%")
                if morb_risk >= 0.30:
                    st.error("⚠️ High risk of major complications. Consider step-down unit admission and targeted pre-habilitation.")
                elif morb_risk >= 0.20:
                    st.warning("Moderate risk of major complications. Ensure strict post-operative monitoring.")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}. Please ensure the input data formats match the training data exactly.")
# --- MODEL PERFORMANCE DETAILS ---
st.markdown("---")
with st.expander("ℹ️ Model Performance & Methodology Details"):
    st.markdown("""
    ### Methodology
    These predictive models were developed using the American College of Surgeons National Surgical Quality Improvement Program (ACS-NSQIP) database.
    American College of Surgeons National Surgical Quality Improvement Program and the hospitals participating in the ACS NSQIP are the source of the data used herein; they     have not verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors

    ### Performance Metrics (Validation Set)
    
    **30-Day Mortality (Random Forest Model)**
    * **AUC (C-statistic):** 0.931 

    **30-Day Major Morbidity (XGBoost Model)**
    * **AUC (C-statistic):** 0.771 
    *
    """)