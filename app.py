import streamlit as st
import joblib, json, pandas as pd

st.title("Bank Term Deposit Prediction")

with open("artifacts/best_model.json") as f:
    best = json.load(f)["best_model"]

models = {
    "logistic":"Logistic Regression",
    "random_forest":"Random Forest",
    "gradient_boost":"Gradient Boosting",
    "xgboost":"XGBoost",
    "lightgbm":"LightGBM",
    "svm":"SVM",
    "naive_bayes":"Naive Bayes",
    "mlp":"Neural Network",
    "stacked_ensemble":"Stacked Ensemble"
}

options = [
    f"{v} (BEST)" if k==best else v
    for k,v in models.items()
]

choice = st.selectbox("Select Model", options)
model_key = [k for k,v in models.items() if v in choice][0]

model = joblib.load(f"artifacts/models/{model_key}.joblib")

st.write("Enter Inputs:")
inputs = {col: st.text_input(col) for col in model.feature_names_in_}

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    prob = model.predict_proba(df)[0][1]
    st.success(f"Subscription Probability: {prob:.2%}")
