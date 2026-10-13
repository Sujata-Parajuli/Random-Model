import streamlit as st
import joblib
import numpy as np
import os

# Get base path to load .pkl model files correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "UCS_model.pkl")
scaler_path = os.path.join(BASE_DIR, "UCS_scaler.pkl")

# Load trained model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# App title
st.title("UCS Prediction for Cemented Mine Backfill")
st.write("Predict Uniaxial Compressive Strength of cemented tailings backfill")
st.divider()

# Input features (Backfill parameters: SD, AT, CD, CT)
sd = st.number_input("Solid Content (SD, %)", min_value=0.0, max_value=100.0, value=72.0)
at = st.number_input("Apparent Total Porosity (AT, %)", min_value=0.0, max_value=50.0, value=12.0)
cd = st.number_input("Cement Dosage (CD, %)", min_value=0.0, max_value=20.0, value=4.0)
ct = st.number_input("Curing Time (CT, days)", min_value=0.0, value=28.0)

# Combine inputs (ORDER MUST MATCH your training dataset: SD, AT, CD, CT)
input_data = np.array([[sd, at, cd, ct]])

# Prediction button
if st.button("Predict UCS"):
    input_scaled = scaler.transform(input_data)
    ucs_pred = model.predict(input_scaled)
    st.success(f"Predicted UCS = {ucs_pred[0]:.2f} MPa")

# Show your input parameters
st.divider()
st.subheader("Input Backfill Parameters")
st.write({
    "SD (Solid Content %)": sd,
    "AT (Apparent Total Porosity %)": at,
    "CD (Cement Dosage %)": cd,
    "CT (Curing Time, days)": ct
})


