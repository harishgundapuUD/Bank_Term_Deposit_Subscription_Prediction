import numpy as np
import os
import json
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from lightgbm import LGBMClassifier

print("===========================================================================")
print("Training STACKED MODEL (RF + LightGBM) with Stratified K-Fold CV...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "roc_auc": []
}

# ----------------------------
# Define base + meta models
# ----------------------------
base_models = [
    ("rf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )),
    ("lgbm", LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1
    ))
]


meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    stack_method="predict_proba",
    cv=5,
    n_jobs=-1
)

config = {}
with open("utils/config.json", "r") as f:
    config = json.load(f)

if config:
        TARGET_COL = config["target_column"]
        drop_cols = config.get("target_column", [])

MODEL_DIR = os.path.join(
                            config["ml_model_dirs"]["base_dir"],
                            config["ml_model_dirs"]["ensembel_model"]
                        )
mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
os.makedirs(mlruns_path, exist_ok=True)
model_metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
mlflow.set_experiment("ensembel_model")

mlflow.end_run()  # safety reset (only if needed)

df = pd.read_csv("datasets/cleaned_train.csv")  # replace with your CSV file path

X = df.drop(columns=drop_cols)
y = df[TARGET_COL]

best_score = -1
best_model_name = None

results = {
                    "ensembel_model": {}
            }

with mlflow.start_run(run_name="stacked_rf_lgbm") as run:

    run_id = run.info.run_id

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Fit stacking model
        stack_model.fit(X_train, y_train)

        y_pred = stack_model.predict(X_val)

        # Probabilities for ROC-AUC
        y_prob = stack_model.predict_proba(X_val)[:, 1]

        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_prob)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)
        fold_metrics["roc_auc"].append(roc_auc)

    # ----------------------------
    # Average across folds
    # ----------------------------
    accuracy = np.mean(fold_metrics["accuracy"])
    precision = np.mean(fold_metrics["precision"])
    recall = np.mean(fold_metrics["recall"])
    f1 = np.mean(fold_metrics["f1"])
    roc_auc = np.mean(fold_metrics["roc_auc"])

    # ----------------------------
    # MLflow logging
    # ----------------------------
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # ----------------------------
    # Train final model on full data
    # ----------------------------
    stack_model.fit(X, y)
    mlflow.sklearn.log_model(stack_model, "stacked_model")

    # ----------------------------
    # Custom score (same as yours)
    # ----------------------------
    score = (
                0.4 * precision +
                0.3 * roc_auc +
                0.2 * f1 +
                0.1 * accuracy
            )

    print(f"STACKED MODEL SCORE: {score:.4f}")

    results["ensembel_model"] = {
                                    "run_id": run_id,
                                    "accuracy": float(accuracy),
                                    "precision": float(precision),
                                    "recall": float(recall),
                                    "f1-score": float(f1),
                                    "roc-auc": float(roc_auc),
                                    "final_score": float(score),
                                    "bestmodel": "no"
                                }

    # update best model if needed
    if score > best_score:
        best_score = score
        best_model_name = "stacked_model"
    
    # save JSON
    with open(model_metrics_path, "w") as f:
        json.dump(results, f, indent=4)

print("===========================================================================")