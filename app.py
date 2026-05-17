import streamlit as st
import pandas as pd
import mlflow.sklearn
import json
import os
from src.data_cleaning import Preprocessing
from training import MODEL_DIR

# -----------------------------
# Load MLflow Model
# -----------------------------
config = {}
with open("utils/config.json", "r") as f:
    config = json.load(f)
MODEL_URI = "models:/your_model_name/Production"

@st.cache_resource
def load_model():
    MODEL_DIR = os.path.join(
                            config["ml_model_dirs"]["base_dir"],
                            config["ml_model_dirs"]["ensembel_model"]
                        )
    mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
    with open("models/ensembel_model/model_metrics.json", "r") as f:
        model_metrics = json.load(f)
    run_id = model_metrics["ensembel_model"]["run_id"]
    MODEL_URI = f"runs:/{run_id}/stacked_rf_lgbm"
    model = mlflow.sklearn.load_model(MODEL_URI)
    return model

model = load_model()

# -----------------------------
# Streamlit App Title
# -----------------------------
st.title("Bank Marketing Prediction App")

st.write("Enter customer details to get prediction.")

# -----------------------------
# Input Fields
# -----------------------------

# Numeric Inputs
age = st.number_input("Age", min_value=18, value=30)

balance = st.number_input("Balance", value=0.0)

day = st.number_input("Day", min_value=1, max_value=31, value=1)

duration = st.number_input("Duration", min_value=0, value=100)

campaign = st.number_input("Campaign", min_value=0, value=1)

pdays = st.number_input("Pdays", min_value=-1, value=-1)

previous = st.number_input("Previous", min_value=0, value=0)

# Categorical Inputs
job = st.selectbox(
    "Job",
    [
        'technician',
        'blue-collar',
        'student',
        'admin',
        'management',
        'entrepreneur',
        'self-employed',
        'unknown',
        'services',
        'retired',
        'housemaid',
        'unemployed'
    ]
)

marital = st.selectbox(
    "Marital Status",
    ['married', 'single', 'divorced']
)

education = st.selectbox(
    "Education",
    ['unknown', 'primary', 'secondary', 'tertiary']
)

default = st.selectbox(
    "Default",
    [0, 1]
)

housing = st.selectbox(
    "Housing Loan",
    [0, 1]
)

loan = st.selectbox(
    "Personal Loan",
    [0, 1]
)

# Additional categorical fields
contact = st.text_input("Contact", value="cellular")

month = st.text_input("Month", value="may")

poutcome = st.text_input("Poutcome", value="unknown")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome
    }])

    try:
        processed_data = Preprocessing(data=input_data, config=config, save_data=False).process()
        prediction = model.predict(processed_data)

        st.success(f"Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")