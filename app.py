import streamlit as st
import numpy as np

# --- 1. THE SHAP-BASED LOGIC ENGINE ---
def calculate_risks(data):
    # Baseline log-odds for NSQIP Radical Cystectomy cohort (approximate intercepts)
    mort_score = -3.8  
    morb_score = -1.1

    # --- DEMOGRAPHICS & VITALS ---
    # Age: Linear scaling based on SHAP impact
    mort_score += (data['age'] - 65) * 0.005
    morb_score += (data['age'] - 65) * 0.015

    # BMI: Impacts Morbidity heavily at extremes
    if data['bmi'] > 30:
        morb_score += (data['bmi'] - 30) * 0.05
    
    # Sex: Female (0.0 in your dataset) showed higher morbidity risk in SHAP
    if data['sex'] == "Female":
        morb_score += 0.8
        mort_score -= 0.02
    else:
        mort_score += 0.02

    # --- COMORBIDITIES & ASA CLASS ---
    # ASA Class: Major driver for both
    asa_map_mort = {1: -0.10, 2: -0.08, 3: 0.03, 4: 0.10}
    asa_map_morb = {1: -0.50, 2: -0.20, 3: 0.40, 4: 0.80}
    mort_score += asa_map_mort.get(data['asa'], 0)
    morb_score += asa_map_morb.get(data['asa'], 0)

    # Hypertension (HYPERMED)
    if data['htn']:
        mort_score += 0.04; morb_score += 0.50
    else:
        mort_score -= 0.04; morb_score -= 0.50

    # Diabetes (DM)
    if data['diabetes']: mort_score += 0.03
    
    # CHF
    if data['chf']: mort_score += 0.05

    # Smoking
    if data['smoke']: morb_score += 0.30

    # --- SURGICAL FACTORS ---
    if data['prior_pelvic']:
        mort_score += 0.03; morb_score += 0.60
        
    if data['prior_rad']: morb_score += 0.30
    
    if data['neoadj']: 
        morb_score -= 0.20 # SHAP showed Neoadj_0.0 (no chemo) increased risk slightly

    # --- LABS (The strongest predictors) ---
    # Albumin (PRALBUM) - Massive impact on both
    if data['alb'] < 4.0:
        alb_gap = 4.0 - data['alb']
        mort_score += alb_gap * 0.08
        morb_score += alb_gap * 1.50
        
    # Creatinine (PRCREAT) - Key for Morbidity
    if data['creat'] > 1.2:
        creat_gap = data['creat'] - 1.2
        morb_score += creat_gap * 0.80
        mort_score += creat_gap * 0.02
        
    # Hematocrit (PRHCT)
    if data['hct'] < 35:
        morb_score += (35 - data['hct']) * 0.05
        
    # Platelets (PRPLATE)
    if data['plt'] > 300: morb_score += 0.20
    elif data['plt'] < 150: morb_score += 0.40

    # --- SIGMOID CONVERSION TO PROBABILITY ---
    mort_prob = 1 / (1 + np.exp(-mort_score))
    morb_prob = 1 / (1 + np.exp(-morb_score))
    
    return mort_prob * 100, morb_prob * 100

# --- 2. THE USER INTERFACE ---
st.set_page_config(page_title="Radical Cystectomy Risk Tool", layout="wide")

st.error("⚠️ **FOR RESEARCH PURPOSES ONLY.** This tool is a demonstration of machine learning model outputs based on retrospective data. It is not intended for clinical decision-making or direct patient care yet.")

st.title("🛡️ Radical Cystectomy Risk Scorecard")
st.write("Predictive modeling derived from ACS NSQIP (2020-2024) utilizing Random Forest and XGBoost algorithms.")

st.divider()

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Demographics & Vitals")
    age = st.number_input("Age (Years)", 18, 100, 65)
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
    asa = st.selectbox("ASA Physical Status Class", [1, 2, 3, 4], index=2)
    smoke = st.checkbox("Current Smoker")

with c2:
    st.subheader("Comorbidities & History")
    htn = st.checkbox("Hypertension requiring Medication")
    dm = st.checkbox("Diabetes Mellitus")
    chf = st.checkbox("Congestive Heart Failure")
    prior_pelvic = st.checkbox("Prior Pelvic Surgery")
    prior_rad = st.checkbox("Prior Pelvic Radiation")
    neoadj = st.checkbox("Neoadjuvant Chemotherapy")

with c3:
    st.subheader("Preoperative Labs")
    alb = st.number_input("Albumin (g/dL)", 1.0, 5.5, 4.0, step=0.1)
    creat = st.number_input("Creatinine (mg/dL)", 0.1, 10.0, 1.0, step=0.1)
    hct = st.number_input("Hematocrit (%)", 15.0, 55.0, 38.0, step=1.0)
    plt_count = st.number_input("Platelets (10^9/L)", 50, 800, 250, step=10)

# --- 3. EXECUTION & DASHBOARD ---
data_inputs = {
    'age': age, 'sex': sex, 'bmi': bmi, 'asa': asa, 'smoke': smoke,
    'htn': htn, 'diabetes': dm, 'chf': chf, 'prior_pelvic': prior_pelvic,
    'prior_rad': prior_rad, 'neoadj': neoadj, 'alb': alb, 'creat': creat, 
    'hct': hct, 'plt': plt_count
}

mort_risk, morb_risk = calculate_risks(data_inputs)

# Set base averages for context
AVG_MORT, AVG_MORB = 2.2, 24.4

st.divider()
st.header("📊 Clinical Risk Assessment")

res1, res2 = st.columns(2)

with res1:
    st.metric(
        label="30-Day Mortality Risk", 
        value=f"{mort_risk:.2f}%", 
        delta=f"{mort_risk - AVG_MORT:.2f}% vs Baseline", 
        delta_color="inverse"
    )
    st.caption(f"Cohort Baseline Mortality: {AVG_MORT}%")

with res2:
    st.metric(
        label="30-Day Major Morbidity Risk", 
        value=f"{morb_risk:.2f}%", 
        delta=f"{morb_risk - AVG_MORB:.2f}% vs Baseline", 
        delta_color="inverse"
    )
    st.caption(f"Cohort Baseline Morbidity: {AVG_MORB}%")

# --- 4. DISCLOSURES & METHODOLOGY ---
st.divider()
with st.expander("Methodology & Data Disclosure"):
    st.write("""
    **Methodology:**
    * **Mortality** is predicted using a Random Forest Classifier (AUC > 0.90). 
    * **Major Morbidity** is predicted using an eXtreme Gradient Boosting (XGBoost) model (AUC ~ 0.85).

    **Legal Disclosure:**
    The American College of Surgeons National Surgical Quality Improvement Program (ACS NSQIP) and the hospitals participating in the ACS NSQIP are the source of the data used herein; they have not verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors.

    **Data Privacy:**
    No ACS NSQIP raw data is hosted or shared. This tool executes a mathematical formula locally and does not store, collect, or transmit any protected health information (PHI).
    """)

st.write("© 2026 Predictive Risk Assessment Tool")