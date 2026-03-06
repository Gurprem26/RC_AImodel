import streamlit as st
import numpy as np

# --- 1. THE SHAP-BASED LOGIC ENGINE ---
def calculate_risks(data):
    # Starting "Baseline" (Log-odds for ~2.2% mortality and ~24% morbidity)
    mort_score = -4.2 
    morb_score = -1.1

    # --- DEMOGRAPHICS ---
    # Age impact
    age_impact = (data['age'] - 65) * 0.04
    mort_score += age_impact
    morb_score += (data['age'] - 65) * 0.02

    # Race & Ethnicity (Based on SHAP trends)
    if data['race'] == "Black": mort_score += 0.15; morb_score += 0.1
    elif data['race'] == "Asian": mort_score -= 0.1
    
    if data['hispanic'] == "Yes": morb_score += 0.1

    # ASA Class (Major weight from your SHAP plot)
    asa_map_mort = {1: -1.2, 2: -0.6, 3: 0.2, 4: 1.5}
    asa_map_morb = {1: -0.8, 2: -0.4, 3: 0.3, 4: 0.9}
    mort_score += asa_map_mort.get(data['asa'], 0)
    morb_score += asa_map_morb.get(data['asa'], 0)

    # BMI
    if data['bmi'] > 35: morb_score += 0.4
    if data['bmi'] < 18.5: mort_score += 0.3

    # Functional Status
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

    # --- SURGICAL FACTORS ---
    if data['cpt'] == "51550": # Partial (Protective)
        mort_score -= 1.3; morb_score -= 1.6
    if data['neoadj']: mort_score += 0.2
    if data['prior_pelvic']: morb_score += 0.3
    if data['prior_rad']: mort_score += 0.4; morb_score += 0.4

    # --- LABS ---
    # Albumin (Major driver)
    if data['alb'] < 3.5:
        alb_gap = 3.5 - data['alb']
        mort_score += alb_gap * 1.6
        morb_score += alb_gap * 1.1
    
    # Creatinine/Hct/Plt (Minor weights)
    if data['creat'] > 1.5: mort_score += 0.3
    if data['hct'] < 30: mort_score += 0.5; morb_score += 0.4
    if data['plt'] < 150: morb_score += 0.3

    # --- SIGMOID CONVERSION ---
    mort_prob = 1 / (1 + np.exp(-mort_score))
    morb_prob = 1 / (1 + np.exp(-morb_score))
    
    return mort_prob * 100, morb_prob * 100

# --- 2. THE USER INTERFACE ---
st.set_page_config(page_title="Cystectomy Risk Tool", layout="wide")
st.title("🧮 Cystectomy Surgical Risk Scorecard")
st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Demographics")
    age = st.number_input("Age (Years)", 18, 100, 65)
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 26.5)
    race = st.selectbox("Race", ["White", "Black", "Asian", "Native American", "Other"])
    hisp = st.selectbox("Hispanic Ethnicity?", ["No", "Yes"])
    asa = st.selectbox("ASA Physical Status", [1, 2, 3, 4], index=1)
    func = st.radio("Independent Functional Status?", ["Yes", "No"])

with c2:
    st.subheader("Surgical & Labs")
    cpt = st.selectbox("Procedure", ["51590 - Radical Cystectomy", "51550 - Partial Cystectomy"])
    neoadj = st.checkbox("Neoadjuvant Chemotherapy")
    p_pelvic = st.checkbox("Prior Pelvic Surgery")
    p_rad = st.checkbox("Prior Pelvic Radiation")
    st.write("")
    alb = st.number_input("Albumin (g/dL)", 1.0, 5.5, 4.0)
    creat = st.number_input("Creatinine (mg/dL)", 0.1, 10.0, 1.0)
    hct = st.number_input("Hematocrit (%)", 15.0, 55.0, 38.0)
    plt = st.number_input("Platelets (x10^9/L)", 50, 600, 250)

with c3:
    st.subheader("Comorbidities")
    htn = st.checkbox("Hypertension (on Meds)")
    smk = st.checkbox("Current Smoker")
    dm = st.checkbox("Diabetes Mellitus")
    copd = st.checkbox("Severe COPD")
    chf = st.checkbox("CHF (within 30 days)")
    arf = st.checkbox("Acute Renal Failure")
    dia = st.checkbox("Currently on Dialysis")
    asc = st.checkbox("Ascites")
    wtl = st.checkbox(">10% Weight Loss (6 months)")

# --- 3. RESULTS ---
data = {
    'age': age, 'race': race, 'hispanic': hisp, 'bmi': bmi, 'asa': asa,
    'independent': func, 'cpt': cpt.split(" -")[0], 'neoadj': neoadj,
    'prior_pelvic': p_pelvic, 'prior_rad': p_rad, 'alb': alb, 'creat': creat,
    'hct': hct, 'plt': plt, 'htn': htn, 'smoke': smk, 'diabetes': dm,
    'copd': copd, 'chf': chf, 'renal_arf': arf, 'dialysis': dia,
    'ascites': asc, 'wt_loss': wtl
}

mort, morb = calculate_risks(data)

st.markdown("---")
res1, res2 = st.columns(2)
res1.metric("30-Day Mortality Risk", f"{mort:.2f}%")
res2.metric("30-Day Major Morbidity Risk", f"{morb:.2f}%")

if mort < 2.2: st.success("Risk Status: BELOW COHORT AVERAGE")
else: st.error("Risk Status: ABOVE COHORT AVERAGE")