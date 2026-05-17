import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ----------------------------
# CONFIG
# ----------------------------

config = {}
with open("utils/config.json", "r") as f:
    config = json.load(f)

if config:
        TARGET_COL = config["target_column"]
        model_types = config.get("model_types", ["base_models", "advanced_models"])
        drop_cols = config.get("target_column", [])
        models = config.get("models", {})
        

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv("datasets/cleaned_train.csv")  # replace with your CSV file path

X = df.drop(columns=drop_cols)
y = df[TARGET_COL]

# ----------------------------
# MODELS
# ----------------------------
all_models = {
            "base_models": {
                                "logisticregression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                "randomforest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                            },
            "advanced_models": {
                                    "XGBoost": XGBClassifier(
                                                                n_estimators=100,
                                                                max_depth=6,
                                                                learning_rate=0.05,
                                                                objective="binary:logistic",
                                                                eval_metric="logloss"
                                                            ),
                                    "light_gbm": LGBMClassifier(
                                                                    n_estimators=200,
                                                                    learning_rate=0.05,
                                                                    max_depth=-1
                                                                ),
                                    # "GradientBoosting": GradientBoostingClassifier(),
                                    # "svm": LinearSVC(),
                                    # "naive_bayes": GaussianNB(),
                                    # "DeepLearning": Sequential([
                                    #                                 Dense(64, activation='relu', input_shape=(X.shape[1],)),
                                    #                                 Dense(32, activation='relu'),
                                    #                                 Dense(1, activation='sigmoid')  # binary output (0/1)
                                    #                             ])
                                }
    }

# for deep learning, we need to compile and train separately
# # Compile
# dl_model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# # Train
# dl_model.fit(
#     X_train, y_train,
#     epochs=10,
#     batch_size=256,
#     validation_split=0.2,
#     verbose=1
# )

# ----------------------------
# EVALUATION STORAGE
# ----------------------------


# ----------------------------
# MLflow Setup
# ----------------------------

for model_type, models in all_models.items():
    MODEL_DIR = os.path.join(
                config["ml_model_dirs"]["base_dir"],
                config["ml_model_dirs"][model_type]
            )
    model_metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    results = {
                    "models": {model_type: {}}
                }

    if os.path.exists(os.path.join(MODEL_DIR, "model_metrics.json")):
        with open(os.path.join(MODEL_DIR, "model_metrics.json"), "r") as f:
            existing_results = json.load(f)
        results = results | existing_results  # Merge with existing results
    best_model_name = None
    best_score = -1
    mlruns_path = os.path.abspath(os.path.join(MODEL_DIR, "mlruns"))
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")
    mlflow.set_experiment(model_type)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print("===========================================================================")
        print(f"Training {name} with Stratified K-Fold CV...")

        fold_metrics = {
                            "accuracy": [],
                            "precision": [],
                            "recall": [],
                            "f1": [],
                            "roc_auc": []
                        }

        mlflow.end_run()  # safety reset
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                # ROC-AUC (binary safe)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_val)[:, 1]
                    roc_auc = roc_auc_score(y_val, y_prob)
                else:
                    roc_auc = 0.0

                acc = accuracy_score(y_val, y_pred)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)

                fold_metrics["accuracy"].append(acc)
                fold_metrics["precision"].append(prec)
                fold_metrics["recall"].append(rec)
                fold_metrics["f1"].append(f1)
                fold_metrics["roc_auc"].append(roc_auc)

            # ----------------------------
            # AVG ACROSS FOLDS
            # ----------------------------
            accuracy = np.mean(fold_metrics["accuracy"])
            precision = np.mean(fold_metrics["precision"])
            recall = np.mean(fold_metrics["recall"])
            f1 = np.mean(fold_metrics["f1"])
            roc_auc = np.mean(fold_metrics["roc_auc"])

            # ----------------------------
            # LOG TO MLflow
            # ----------------------------
            mlflow.log_metric(f"{name}_accuracy", accuracy)
            mlflow.log_metric(f"{name}_precision", precision)
            mlflow.log_metric(f"{name}_recall", recall)
            mlflow.log_metric(f"{name}_f1_score", f1)
            mlflow.log_metric(f"{name}_roc_auc", roc_auc)

            # ----------------------------
            # FINAL TRAIN (FULL DATA)
            # ----------------------------
            model.fit(X, y)
            mlflow.sklearn.log_model(model, name)

            # ----------------------------
            # CUSTOM SCORE
            # ----------------------------
            score = (
                        0.4 * precision +
                        0.3 * roc_auc +
                        0.2 * f1 +
                        0.1 * accuracy
                    )

            print(f"{name} CV Score: {score:.4f}")

            # store results
            results["models"][model_type][name] = {
                                                        "run_id": run_id,
                                                        "accuracy": float(accuracy),
                                                        "precision": float(precision),
                                                        "recall": float(recall),
                                                        "f1-score": float(f1),
                                                        "roc-auc": float(roc_auc),
                                                        "bestmodel": "no"
                                                    }

            # best model tracking
            if score > best_score:
                best_score = score
                best_model_name = name

        print("===========================================================================")
        print("\n")

    # mark best model
    if best_model_name:
        results["models"][model_type][best_model_name]["bestmodel"] = "yes"

    # save JSON
    with open(model_metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Training complete.")
    print("Best model:", best_model_name)