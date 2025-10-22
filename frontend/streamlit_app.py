# frontend/streamlit_app.py
import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("Iris classifier (Frontend)")

st.write("Enter the 4 features to get an Iris prediction:")

sepal_len = st.number_input("sepal length (cm)", 0.0, 10.0, 5.1)
sepal_wid = st.number_input("sepal width (cm)", 0.0, 10.0, 3.5)
petal_len = st.number_input("petal length (cm)", 0.0, 10.0, 1.4)
petal_wid = st.number_input("petal width (cm)", 0.0, 10.0, 0.2)

if st.button("Predict"):
    payload = {"features": [sepal_len, sepal_wid, petal_len, petal_wid]}
    try:
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            st.error(f"Backend error: {data}")
        else:
            st.success(f"Prediction: {data['prediction']}")
            if data.get("probability"):
                st.write("Probabilities:", data["probability"])
    except Exception as e:
        st.error(f"Request failed: {e}")
