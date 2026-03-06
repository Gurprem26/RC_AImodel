# C_AImodel
# Cystectomy 30-Day Risk Calculator

## Overview
This repository contains the deployment code and trained machine learning models for predicting 30-day post-operative outcomes following radical and partial cystectomy. The models were developed and internally validated using the American College of Surgeons National Surgical Quality Improvement Program (ACS-NSQIP) database.

## Models Deployed
* **30-Day Mortality:** Calibrated Random Forest Classifier
* **30-Day Major Morbidity:** Calibrated XGBoost Classifier


## Files in this Repository
* `app.py`: The Streamlit web application script.
* `requirements.txt`: Python environment dependencies.
* `preprocessor.pkl`: The serialized data preprocessor (handles scaling and One-Hot Encoding).
* `mortality_rf_model.pkl`: The serialized Random Forest model.
* `morbidity_xgb_model.pkl`: The serialized XGBoost model.

## Data Availability
**⚠️ Note:** The raw patient dataset derived from ACS-NSQIP is not included in this repository or shared anywhere to comply with the ACS-NSQIP Data Use Agreement (DUA) and patient privacy regulations. 

## Research Use Only
This tool is strictly for academic and research purposes. It is not intended to diagnose, treat, cure, or prevent any disease, nor should it replace the clinical judgment of a licensed healthcare professional.
