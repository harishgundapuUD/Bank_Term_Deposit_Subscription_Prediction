import joblib, shap, pandas as pd

model = joblib.load("artifacts/models/random_forest.pkl")
X = pd.read_csv("datasets/train.csv").drop(columns=["y"])

explainer = shap.Explainer(model.named_steps["model"])
X_transformed = model.named_steps["preprocess"].transform(
    model.named_steps["interaction"].transform(X)
)


shap_values = explainer(X_transformed)
shap.summary_plot(shap_values, X_transformed)
