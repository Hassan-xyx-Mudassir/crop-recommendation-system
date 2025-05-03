# app.py
import streamlit as st
import joblib
import numpy as np
import os

# Load model
MODEL_PATH = os.path.join('saved_models', 'rf_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load("saved_models/label_encoder.pkl")

# Page config
st.set_page_config(page_title="Crop Recommendation", layout="centered")

st.title("🌾 Crop Recommendation System")

st.markdown("Please fill in all the environmental and soil parameters below:")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0, step=1.0)
P = st.number_input("Phosphorous (P)", min_value=0.0, step=1.0)
K = st.number_input("Potassium (K)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity (%)", format="%.2f")
ph = st.number_input("pH Level", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")

# Button
if st.button("Recommend"):
    # Check if any field is left empty (all must be > 0 or > reasonable threshold)
    inputs = [N, P, K, temperature, humidity, ph, rainfall]
    if any(val is None or val == 0.0 for val in inputs):
        st.error("🚨 Please fill in all fields with valid values.")
    else:
        input_features = np.array(inputs).reshape(1, -1)
        prediction_encoded = model.predict(input_features)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        st.success(f"✅ Recommended Crop: **{prediction_label.capitalize()}**")
