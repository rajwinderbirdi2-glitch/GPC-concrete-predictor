# concrete_predictor_streamlit.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# --- Dummy model for demonstration ---
# Replace this with your trained model if you have one
model = LinearRegression()
# Example training data (just for demonstration)
X_train = np.array([
    [120, 0, 525, 514, 76, 30, 11, 22, 43, 2, 13, 3, 0, 0, 56, 0, 24, 0, 24],
    [530, 360, 1277, 811, 196, 103, 29, 58, 113, 16, 74, 38, 170, 18, 304, 1, 100, 2, 30]
])
y_train = np.array([20, 50])  # Replace with actual strengths
model.fit(X_train, y_train)

# --- Streamlit App ---
st.title("Concrete Strength Predictor")

st.subheader("Input Mix Parameters")

# List of features
features = [
    "FA (kg/m3)", "GGBFS (kg/m3)", "Coarse aggregate (kg/m3)", "Fine aggregate (kg/m3)",
    "Na2SiO3", "NaOH", "Na2O (Dry)", "Sio2 (Dry)", "Water (1)",
    "Concentration (M) NaOH", "Water (2)", "NaOH (Dry)", "Additional water",
    "Superplasticizer", "Total water", "Initial curing time (day)",
    "Initial curing temp (C)", "Initial curing rest time (day)", "Final curing temp (C)"
]

# Default values from your pasted data
defaults = [
    255, 165, 991, 744, 149, 60, 18, 45, 86, 8, 40, 20, 21, 12, 154, 0, 37, 1, 28
]

# Input sliders
inputs = []
for feature, default in zip(features, defaults):
    value = st.number_input(feature, value=float(default))
    inputs.append(value)

# Predict button
if st.button("Predict"):
    X_input = np.array(inputs).reshape(1, -1)
    pred = model.predict(X_input)
    st.success(f"Predicted Concrete Strength: {pred[0]:.2f} MPa")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(features))
    ax.bar(x, inputs, color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("Concrete Mix Parameters")
    st.pyplot(fig)
