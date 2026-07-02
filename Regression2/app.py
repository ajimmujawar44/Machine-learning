import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("House_Prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page title
st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Predict house price using Living Area (sqft).")

# Sidebar
st.sidebar.header("Input")

sqft = st.sidebar.number_input(
    "Living Area (sqft)",
    min_value=500,
    max_value=15000,
    value=2000,
    step=50
)

# Prediction button
if st.button("Predict Price"):

    prediction = model.predict([[sqft]])

    st.success(f"Estimated House Price: ₹ {prediction[0]:,.2f}")

st.write("---")
st.write("Developed using Streamlit & Scikit-Learn")