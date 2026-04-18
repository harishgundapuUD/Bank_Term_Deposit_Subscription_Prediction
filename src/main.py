import os, json, joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -------------------- CONFIG --------------------
ARTIFACT_DIR = "artifacts"
MODEL_DIR = f"{ARTIFACT_DIR}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "y"

# -------------------- FEATURE ENGINEERING --------------------
class InteractionAdder:
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        if {"duration","balance"}.issubset(X.columns):
            X["duration_balance"] = X["duration"] * X["balance"]
        if {"age","balance"}.issubset(X.columns):
            X["age_balance"] = X["age"] * X["balance"]
        return X

# -------------------- LOAD DATA --------------------
train = pd.read_csv("datasets/train.csv")
X = train.drop(columns=[TARGET])
y = train[TARGET]

num_cols = X.select_dtypes(include=["int64","float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

# -------------------- MODELS --------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=300),
    "gradient_boost": GradientBoostingClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss"),
    "lightgbm": LGBMClassifier(force_col_wise=True),
    "svm": SVC(probability=True),
    "naive_bayes": GaussianNB(),
    "mlp": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# -------------------- TRAIN & EVALUATE --------------------
for name, model in models.items():
    pipeline = Pipeline([
        ("interaction", InteractionAdder()),
        ("preprocess", preprocessor),
        ("model", model)
    ])

    metrics = {"accuracy":[], "precision":[], "recall":[], "f1":[], "roc_auc":[]}

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        probs = pipeline.predict_proba(X_val)[:,1]

        metrics["accuracy"].append(accuracy_score(y_val, preds))
        metrics["precision"].append(precision_score(y_val, preds))
        metrics["recall"].append(recall_score(y_val, preds))
        metrics["f1"].append(f1_score(y_val, preds))
        metrics["roc_auc"].append(roc_auc_score(y_val, probs))

    results[name] = {k: np.mean(v) for k,v in metrics.items()}
    pipeline.fit(X, y)
    joblib.dump(pipeline, f"{MODEL_DIR}/{name}.pkl")

# -------------------- STACKING ENSEMBLE --------------------
estimators = [(k, joblib.load(f"{MODEL_DIR}/{k}.pkl")) for k in models]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack_pipe = Pipeline([
    ("interaction", InteractionAdder()),
    ("preprocess", preprocessor),
    ("model", stack)
])

stack_pipe.fit(X, y)
joblib.dump(stack_pipe, f"{MODEL_DIR}/stacked_ensemble.pkl")

# -------------------- SAVE METRICS --------------------
df = pd.DataFrame(results).T.sort_values("roc_auc", ascending=False)
df.to_csv(f"{ARTIFACT_DIR}/model_metrics.csv")

best_model = df.index[0]
with open(f"{ARTIFACT_DIR}/best_model.json","w") as f:
    json.dump({"best_model":best_model}, f, indent=2)

print("✅ Training complete. All models saved.")
