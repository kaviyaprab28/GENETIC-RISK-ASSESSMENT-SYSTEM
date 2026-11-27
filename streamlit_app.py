
import json
import requests
import pandas as pd
import streamlit as st

# Page setup
st.set_page_config(
    page_title="Genetic Disorder Risk Assessment",
    page_icon="🧬",
    layout="wide"
)

API_URL = "https://genetic-risk-assessment-api.onrender.com/predict"

st.title("🧬 Genetic Disorder Risk Assessment System")
st.write(
    "Enter the patient details below to predict the genetic disorder category. "
    "This is an educational ML project, not a medical diagnosis."
)

# Load input fields and dropdown choices
with open("input_schema.json", "r") as file:
    schema = json.load(file)

# Patient input form
with st.form("patient_form"):
    st.subheader("Patient Details")

    patient_data = {}
    columns = st.columns(2)

    for index, (field, details) in enumerate(schema.items()):
        with columns[index % 2]:
            if details["type"] == "number":
                patient_data[field] = st.number_input(
                    field,
                    value=float(details["default"])
                )
            else:
                options = details["options"]
                patient_data[field] = st.selectbox(
                    field,
                    options=options,
                    index=options.index(details["default"])
                    if details["default"] in options else 0
                )

    submitted = st.form_submit_button("Predict Disorder")

# Send input to Flask API
if submitted:
    try:
        response = requests.post(API_URL, json=patient_data, timeout=30)
        response.raise_for_status()
        result = response.json()

        prediction = result.get(
            "Predicted Disorder",
            result.get("prediction", result.get("predicted_disorder", result.get("disorder", result)))
        )

        st.subheader("Predicted Disorder")
        if isinstance(prediction, dict):
            prediction = prediction.get(
                "Predicted Disorder",
                prediction.get("prediction", prediction.get("disorder", str(prediction)))
            )

        st.success(str(prediction))

    except requests.exceptions.RequestException as error:
        st.error(
            "Could not connect to the Flask API. "
            "Check that the Flask server is running."
        )
        st.caption(str(error))

# Feature importance section
st.divider()
st.subheader("Important Features")

try:
    importance = pd.read_csv("feature_importance.csv")
    st.dataframe(importance.head(10), use_container_width=True)
except Exception:
    st.info("Feature importance file is not available.")

if "feature_importance.png" in __import__("os").listdir():
    st.image("feature_importance.png", caption="Feature Importance")

# Decision Tree visualization
st.divider()
st.subheader("Decision Tree Visualization")

if "decision_tree.png" in __import__("os").listdir():
    st.image("decision_tree.png", caption="Decision Tree (visualized to depth 3)")
else:
    st.info("Decision Tree image is not available.")

# Model performance
st.divider()
st.subheader("Model Performance")

st.write("Tuned Decision Tree Classifier")
col1, col2, col3 = st.columns(3)

col1.metric("Test Accuracy", "60.08%")
col2.metric("Macro F1-score", "0.51")
col3.metric("Train-Test Gap", "3.03 percentage points")
