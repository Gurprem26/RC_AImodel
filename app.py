import streamlit as st
import numpy as np

# --- 1. THE SHAP-BASED LOGIC ENGINE ---
def calculate_risks(data):
    # Adjusted Baseline for a Radical Cystectomy-only cohort
    mort_score = -4.1 
    morb_score = -1.0

    # --- DEMOGRAPHICS & ASA ---
    age_impact = (data['age'] - 65) * 0.04
    mort_score += age_impact
    morb_score += (data['age'] - 65) * 0.02

    if data['race'] == "Black": mort_score += 0.15; morb_score += 0.1
    if data['hispanic'] == "Yes": morb_score += 0.1

    # ASA Class: Major driver
    asa_map_mort = {1: -1.3, 2: -0.7, 3: 0.3, 4: 1.6}
    asa_map_morb = {1: -0.9, 2: -0.5, 3: 0.4, 4: 1.0}
    mort_score += asa_map_mort.get(data['asa'], 0)
    morb_score += asa_map_morb.get(data['asa'], 0)

    if data['bmi'] > 35: morb_score += 0.4
    if data['bmi'] < 18.5: mort_score += 0.3
    if data['independent'] == "No": 
        mort_score += 0.8; morb_score += 0.6

    # --- MEDICAL COMORBIDITIES ---
    if data['htn']: morb_score += 0.2
    if data['smoke']: mort_score += 0.2; morb_score += 0.3
    if data['diabetes']: mort_score += 0.3; morb_score += 0.4
    if data['copd']: mort_score += 0.6; morb_score += 0.4
    if data['chf']: mort_score += 1.3; morb_score += 0.9
    if data['renal_arf']: mort_score += 1.0; morb_score += 0.8
    if data['dialysis']: mort_score += 1.2; morb_score += 0.8
    if data['ascites']: mort_score += 1.6; morb_score += 1.1
    if data['wt_loss']: mort_score += 0.7; morb_score += 0.5

    # --- SURGICAL FACTORS (Radical Only) ---
    if data['procedure'] == "Radical Cystectomy w/ Neobladder":
        # Neobladder typically has slightly higher morbidity due to operative time/complexity
        mort_score += 0.05; morb_score += 0.4
    else:
        # Ileal Conduit is the standard baseline
        mort_score += 0.0; morb_score += 0.0

    if data['neoadj']: mort_score += 0.2
    if data['prior_rad']: mort_score += 0.4; morb_score += 0.4

    # --- LABS ---
    if data['alb'] < 3.5:
        alb_gap = 3.5 - data['alb']
        mort_score += alb_gap * 1.6
        morb_score += alb_gap * 1.1
    
    if data['hct'] < 30: 
        mort_score += 0.5; morb_score += 0.4

    # --- SIGMOID CONVERSION ---
    mort_prob = 1 / (1 + np.exp(-mort_score))
    morb_prob = 1 / (1 + np.exp(-morb_score))
    
    return mort_prob * 100, morb_prob * 100

# --- 2. THE USER INTERFACE ---
st.set_page_config(page_title="Radical Cystectomy Risk Tool", layout="wide")

st.error("⚠️ **FOR RESEARCH PURPOSES ONLY.** This tool is not for clinical decision-making or direct patient care. All estimates must be validated by a clinical professional.")

st.title("🧮 Radical Cystectomy Risk Scorecard")
st.write("Comparing individual risk to NSQIP population averages (2020-2024).")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Demographics")
    age = st.number_input("Age (Years)", 18, 100, 65)
    bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
    hisp = st.selectbox("Hispanic Ethnicity?", ["No", "Yes"])
    asa = st.selectbox("ASA Physical Status", [1, 2, 3, 4], index=1)
    func = st.radio("Independent Functional Status?", ["Yes", "No"])

with c2:
    st.subheader("Surgical & Labs")
    proc = st.selectbox("Procedure Type", [
        "Radical Cystectomy w/ Ileal Conduit", 
        "Radical Cystectomy w/ Neobladder"
    ])
    neoadj = st.checkbox("Neoadjuvant Chemotherapy")
    p_rad = st.checkbox("Prior Pelvic Radiation")
    st.write("")
    alb = st.number_input("Albumin (g/dL)", 1.0, 5.5, 4.0)
    hct = st.number_input("Hematocrit (%)", 15.0, 55.0, 38.0)

with c3:
    st.subheader("Comorbidities")
    htn = st.checkbox("Hypertension")
    smk = st.checkbox("Current Smoker")
    dm = st.checkbox("Diabetes Mellitus")
    copd = st.checkbox("Severe COPD")
    chf = st.checkbox("CHF")
    arf = st.checkbox("Acute Renal Failure")
    dia = st.checkbox("On Dialysis")
    asc = st.checkbox("Ascites")
    wtl = st.checkbox(">10% Weight Loss")

# --- 3. EXECUTION & DASHBOARD ---
data = {
    'age': age, 'race': race, 'hispanic': hisp, 'bmi': bmi, 'asa': asa,
    'independent': func, 'procedure': proc, 'neoadj': neoadj,
    'prior_rad': p_rad, 'alb': alb, 'hct': hct, 'htn': htn, 
    'smoke': smk, 'diabetes': dm, 'copd': copd, 'chf': chf, 
    'renal_arf': arf, 'dialysis': dia, 'ascites': asc, 'wt_loss': wtl
}

mort, morb = calculate_risks(data)
AVG_MORT, AVG_MORB = 2.2, 24.4

st.divider()
st.header("📊 Clinical Risk Assessment")
res1, res2 = st.columns(2)

with res1:
    st.metric("30-Day Mortality", f"{mort:.2f}%", delta=f"{mort - AVG_MORT:.2f}% vs Avg", delta_color="inverse")
    st.caption(f"Cohort Average Mortality: {AVG_MORT}%")
    if mort < AVG_MORT: st.success("Lower than average mortality risk.")
    else: st.warning("Above average mortality risk.")

with res2:
    st.metric("30-Day Major Morbidity", f"{morb:.2f}%", delta=f"{morb - AVG_MORB:.2f}% vs Avg", delta_color="inverse")
    st.caption(f"Cohort Average Morbidity: {AVG_MORB}%")
    if morb < AVG_MORB: st.success("Lower than average morbidity risk.")
    else: st.warning("Above average morbidity risk.")

# --- 4. DISCLOSURES & LEGAL ---
st.divider()
with st.expander("Methodology & Data Disclosure"):
    st.write("""
    **Methodology:**
    This tool was developed using machine learning discovery. We utilized **Random Forest** classification to determine predictors for **30-Day Mortality** and **XGBoost** for **30-Day Major Morbidity**. Risk weights were extracted via SHAP (SHapley Additive exPlanations) to ensure a transparent scoring system.

    **Legal Disclosure:**
    American College of Surgeons National Surgical Quality Improvement Program (ACS NSQIP) and the hospitals participating in the ACS NSQIP are the source of the data used herein; they have not verified and are not responsible for the statistical validity of the data analysis or the conclusions derived by the authors.

    **Data Privacy:**
    ACS NSQIP data is not shared here or with anyone. This application does not store or transmit patient-identifiable information.
    """)

st.write("© 2026 Research Validation Tool | Radical Cystectomy ML Project")