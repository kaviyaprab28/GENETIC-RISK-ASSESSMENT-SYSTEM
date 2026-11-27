# GENETIC-RISK-ASSESSMENT-SYSTEM
Explainable Genetic Disorder Risk Assessment System using Decision Tree, Flask and Streamlit.
# Genetic Disorder Risk Assessment System

An Explainable Machine Learning project that predicts genetic disorder categories using a Decision Tree Classifier.

## Project Overview

This project uses patient-related features to predict a genetic disorder category. It also provides feature importance and a Decision Tree visualization to help understand the model's predictions.

This is an educational project and is not intended for medical diagnosis.

## Features

- Genetic disorder category prediction
- Decision Tree classification model
- Feature importance analysis
- Decision Tree visualization
- Flask API for prediction
- Streamlit user interface
- Model performance display

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Flask
- Streamlit
- Joblib

## Project Files

- `streamlit_app.py` – Streamlit frontend
- `flask_api.py` – Flask prediction API
- `genetic_disorder_model.joblib` – Trained model
- `genetic_disorder_preprocessor.joblib` – Data preprocessing pipeline
- `input_schema.json` – Input field definitions
- `feature_importance.csv` – Feature importance values
- `feature_importance.png` – Feature importance visualization
- `decision_tree.png` – Decision Tree visualization

## Model Performance

- Test Accuracy: 60.08%
- Macro F1-score: 0.51
- Train-Test Accuracy Gap: 3.03 percentage points

## Run Locally

Install the required packages:

```bash
pip install -r requirements.txt
