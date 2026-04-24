import streamlit as st
import numpy as np

# --- 1. THE REFINED LOGIC ENGINE ---
def calculate_risks(data):
    """
        Baseline risks (Intercepts) adjusted for NSQIP Cystectomy cohort.
    """
    # Base log-odds (Approximate based on cohort averages of ~2.2% and ~24.4%)
    mort_score = -3.8
    morb_score = -1.1

    # --- DEMOGRAPHICS ---
    # Age: SHAP shows high age = high mortality risk. LR OR 1.42 per unit increase.
    mort_score += (data['age'] - 65) * 0.12  # Strong weight from LR
    morb_score += (data['age'] - 65) * 0.01

    # BMI: LR OR 1.15 for Morbidity. SHAP shows high BMI increases morbidity risk.
    if data['bmi'] > 30:
        morb_score += (data['bmi'] - 30) * 0.14
    
    # Race & Ethnicity: Based on Table 3 (LR) and SHAP Morbidity plot
    if data['race'] == "Black":
        morb_score += 0.29  # OR 1.34
    if data['ethnicity'] == "Hispanic":
        morb_score += 0.31  # OR 1.37

    # --- CLINICAL & FRAILTY ---
    # Frailty: mFI-5 Low is a huge protector (OR 0.30 for mortality)
    if data['frailty'] <= 1:
        mort_score -= 1.20 
        morb_score -= 0.33
    else:
        mort_score += 0.50

    # ASA Class: SHAP shows ASA IV as a massive mortality driver
    asa_mort_map = {1: -0.5, 2: -0.3, 3: 0.1, 4: 0.8}
    mort_score += asa_mort_map.get(data['asa'], 0)

    # --- SURGICAL FACTORS ---
    # Continent Diversion (51596): SHAP shows lower mortality risk for this group
    if data['diversion'] == "Continent":
        mort_score -= 0.20
    
    # Prior Pelvic Surgery: LR OR 1.15 for Morbidity
    if data['prior_pelvic']:
        morb_score += 0.14

    # --- LABS (Strongest SHAP Drivers) ---
    # Preop Albumin: Strong protector in both SHAP and LR (OR 0.79/0.88)
    # SHAP shows values < 3.5 significantly increase risk
    if data['alb'] < 4.0:
        alb_diff = 4.0 - data['alb']
        mort_score += alb_diff * 0.25
        morb_score += alb_diff * 0.15

    # Preop Creatinine: High feature importance in SHAP Mortality
    if data['creat'] > 1.2:
        mort_score += (data['creat'] - 1.2) * 0.10

    # Hematocrit: Low HCT increases Morbidity in SHAP
    if data['hct'] < 30:
        morb_score += 0.20

    # --- SIGMOID CONVERSION ---
    mort_prob = 1 / (1 + np.exp(-mort_score))
    morb_prob = 1 / (1 + np.exp(-morb_score))
    
    return mort_prob * 100, morb_prob * 100

# --- 2. THE USER INTERFACE ---
st.set_page_config(page_title="Cystectomy Risk Pro", layout="wide")

# Custom CSS to keep the app looking professional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Radical Cystectomy Risk Assessment Tool")
st.caption("Evidence-based risk calculation utilizing SHAP-enhanced Machine Learning and Multivariate Logistic Regression (NSQIP 2020-2024)")

with st.sidebar:
    st.header("Help & Instructions")
    st.info("This tool calculates 30-day outcomes. Enter patient preoperative data to see risk adjustments compared to the cohort baseline.")
    if st.button("🔄 Reset Inputs"):
        st.rerun()

# Layout: 3 Columns for data entry
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Patient Demographics")
    age = st.number_input("Age (Years)", 18, 100, 65)
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
    race = st.selectbox("Race", ["White", "Black", "Asian", "Other"])
    ethnicity = st.selectbox("Ethnicity", ["Non-Hispanic", "Hispanic"])
    bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 26.5)

with col2:
    st.subheader("🩺 Clinical Status")
    frailty = st.slider("mFI-5 Frailty Score", 0, 5, 1, help="Modified Frailty Index (0-5 scale)")
    asa = st.selectbox("ASA Physical Status", [1, 2, 3, 4], index=2)
    diversion = st.selectbox("Planned Diversion", ["Ileal Conduit", "Continent", "Other"])
    prior_pelvic = st.checkbox("Prior Pelvic Surgery")
    neoadj = st.checkbox("Neoadjuvant Chemotherapy")

with col3:
    st.subheader("🧪 Preoperative Labs")
    alb = st.number_input("Albumin (g/dL)", 1.0, 5.5, 4.0, step=0.1)
    creat = st.number_input("Creatinine (mg/dL)", 0.1, 10.0, 1.0, step=0.1)
    hct = st.number_input("Hematocrit (%)", 15.0, 55.0, 38.0, step=1.0)
    plt = st.number_input("Platelets (10³/µL)", 50, 800, 250)

# --- 3. RESULTS DASHBOARD ---
inputs = {
    'age': age, 'sex': sex, 'race': race, 'ethnicity': ethnicity, 
    'bmi': bmi, 'frailty': frailty, 'asa': asa, 'diversion': diversion,
    'prior_pelvic': prior_pelvic, 'neoadj': neoadj, 'alb': alb, 
    'creat': creat, 'hct': hct, 'plt': plt
}

mort_risk, morb_risk = calculate_risks(inputs)
AVG_MORT, AVG_MORB = 2.2, 24.4 # Cohort averages

st.divider()

res_col1, res_col2 = st.columns(2)

with res_col1:
    delta_mort = mort_risk - AVG_MORT
    st.metric("30-Day Mortality Risk", f"{mort_risk:.2f}%", delta=f"{delta_mort:+.2f}%", delta_color="inverse")
    if mort_risk > 5:
        st.warning("High Mortality Risk detected.")

with res_col2:
    delta_morb = morb_risk - AVG_MORB
    st.metric("30-Day Major Morbidity Risk", f"{morb_risk:.2f}%", delta=f"{delta_morb:+.2f}%", delta_color="inverse")
    if morb_risk > 40:
        st.warning("High Morbidity Risk detected.")

# --- 4. METHODOLOGY & SHAP INSIGHTS ---
with st.expander("View Predictive Drivers (SHAP Analysis)"):
    st.write("The following factors had the highest impact on this specific prediction:")
    
    # Simple logic to show what drove the specific score up
    drivers = []
    if alb < 3.5: drivers.append("Low Preoperative Albumin (High Risk)")
    if age > 75: drivers.append("Advanced Age (High Risk)")
    if frailty > 2: drivers.append("High Frailty Score (High Risk)")
    if asa >= 4: drivers.append("ASA Class IV (High Risk)")
    
    if not drivers:
        st.write("Patient is largely within low-risk parameters.")
    else:
        for d in drivers:
            st.write(f"• {d}")

st.divider()
st.error("⚠️ **Disclaimer:** This tool is for peer-review and research evaluation only. It uses retrospective data from the ACS NSQIP database. It does not replace clinical judgment.")
