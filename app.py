import streamlit as st
import pandas as pd
import mlflow.sklearn
import json
import os
import yaml
from src.data_cleaning import Preprocessing

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
    with open("models/ensembel_models/model_metrics.json", "r") as f:
        model_metrics = json.load(f)
    run_id = model_metrics["ensembel_model"]["run_id"]
    experiment_dirs = os.listdir(mlruns_path)

    experiment_id = None

    for exp in experiment_dirs:
        possible_run = os.path.join(
                                        mlruns_path,
                                        exp,
                                        run_id
                                    )
        if os.path.exists(possible_run):
            experiment_id = exp
            break

    if experiment_id is None:
        raise Exception("Run ID not found")

    # -------------------------------------------------
    # Outputs directory
    # -------------------------------------------------

    outputs_dir = os.path.join(
                                    mlruns_path,
                                    experiment_id,
                                    run_id,
                                    "outputs"
                                )

    output_folders = os.listdir(outputs_dir)

    yaml_path = os.path.join(
                                outputs_dir,
                                output_folders[0],
                                "meta.yaml"
                            )

    # -------------------------------------------------
    # Read YAML
    # -------------------------------------------------

    with open(yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    model_id = meta["destination_id"]

    model_path = os.path.join(
                                mlruns_path,
                                experiment_id,
                                # run_id,
                                "models",
                                model_id,
                                "artifacts",
                                # "model.pkl"
                            )
    # MODEL_URI = f"runs:/{run_id}/stacked_rf_lgbm"
    model = mlflow.sklearn.load_model(model_path)
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


col1, col2 = st.columns(2)

with col1:

    # Numeric Inputs
    age = st.number_input("Age", min_value=18, value=30)

    balance = st.number_input("Balance", value=0.0)

    day = st.number_input("Date", min_value=1, max_value=31, value=1)

    duration = st.number_input("Duration", min_value=0, value=100)

    campaign = st.number_input("Campaign", min_value=0, value=1)

    pdays = st.number_input(
        "Previous contact days",
        min_value=-1,
        value=-1
    )

    previous = st.number_input(
        "Previous subscriptions",
        min_value=0,
        value=0
    )

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

with col2:

    marital = st.selectbox(
        "Marital Status",
        ['married', 'single', 'divorced']
    )

    education = st.selectbox(
        "Education",
        ['primary', 'secondary', 'tertiary', 'unknown']
    )

    default = st.selectbox(
        "Default",
        ["no", "yes"]
    )

    housing = st.selectbox(
        "Housing Loan",
        ["no", "yes"]
    )

    loan = st.selectbox(
        "Personal Loan",
        ["no", "yes"]
    )

    poutcome = st.selectbox(
        "Previous outcome",
        ['success', 'failure', 'other', 'unknown']
    )

    contact = st.selectbox(
        "Contact",
        ['cellular', 'telephone', 'unknown']
    )

    month = st.selectbox(
        "Month",
        [
            'jan', 'feb', 'mar', 'apr',
            'may', 'jun', 'jul', 'aug',
            'sep', 'oct', 'nov', 'dec'
        ]
    )

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

    # try:
    processed_data = Preprocessing(data=input_data, config=config, save_data=False).process()
    # Align columns with training data

    processed_data = processed_data.reindex(
                                                columns=model.feature_names_in_,
                                                fill_value=0
                                            )
    print(f"[DEBUG] : The processed data is : {processed_data.columns}")
    prediction = model.predict(processed_data)

    prediction_mapping = {
                                0: "User will not take the Subscription",
                                1: "User will take the Subscription"
                            }
    if prediction[0] == 1:
        st.success(f"Prediction: {prediction_mapping[prediction[0]]}")
    else:
        st.error(f"Prediction: {prediction_mapping[prediction[0]]}")

    # except Exception as e:
    #     st.error(f"Prediction failed: {e}")