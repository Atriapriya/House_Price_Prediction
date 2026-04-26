import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load Model & Scaler ───────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# ── UI ────────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.markdown("Enter the house details below to get an estimated price.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    square_footage   = st.number_input("Square Footage (sq ft)", min_value=200, max_value=10000, value=1500, step=50)
    num_bedrooms     = st.number_input("Number of Bedrooms",      min_value=1,   max_value=10,    value=3,    step=1)
    num_bathrooms    = st.number_input("Number of Bathrooms",     min_value=1,   max_value=10,    value=2,    step=1)
    garage_size      = st.number_input("Garage Size (cars)",      min_value=0,   max_value=5,     value=1,    step=1)

with col2:
    neighborhood_quality = st.slider("Neighborhood Quality Score", min_value=1,  max_value=10, value=5)
    year_built           = st.number_input("Year Built",           min_value=1900, max_value=2024, value=2000, step=1)
    distance_to_city     = st.number_input("Distance to City Center (miles)", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
    house_age            = 2024 - year_built

st.divider()

if st.button("🔍 Predict House Price", use_container_width=True, type="primary"):
    # Build input — adjust column order to match your training features
    features = np.array([[
        square_footage,
        num_bedrooms,
        num_bathrooms,
        garage_size,
        neighborhood_quality,
        year_built,
        distance_to_city,
        house_age
    ]])

    features_scaled = scaler.transform(features)
    log_price       = model.predict(features_scaled)[0]
    price           = np.expm1(log_price)  # inverse log transform

    st.success(f"### 💰 Estimated House Price: **${price:,.0f}**")

    st.markdown("---")
    st.markdown("**Input Summary**")
    summary = {
        "Square Footage": square_footage,
        "Bedrooms": num_bedrooms,
        "Bathrooms": num_bathrooms,
        "Garage Size": garage_size,
        "Neighborhood Quality": neighborhood_quality,
        "Year Built": year_built,
        "Distance to City (miles)": distance_to_city,
    }
    st.dataframe(pd.DataFrame(summary.items(), columns=["Feature", "Value"]), hide_index=True)

st.markdown("---")
st.caption("Model: Linear Regression | Trained on house_price_regression_dataset.csv")