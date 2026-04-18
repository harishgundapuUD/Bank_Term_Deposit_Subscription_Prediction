from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
import numpy as np
import pandas as pd
from datetime import datetime

data = pd.read_csv(r"../datasets/train.csv")

education_order = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
data["education"] = data["education"].map(education_order)
data['job'] = data['job'].str.replace('.', '', regex=False)
data = pd.get_dummies(data, columns=['marital', "contact", "poutcome", "job"], dtype="int")
data["default"] = data["default"].map({"no": 0, "yes": 1})
data["housing"] = data["housing"].map({"no": 0, "yes": 1})
data["loan"] = data["loan"].map({"no": 0, "yes": 1})
data["month"] = data["month"].apply(lambda x: datetime.strptime(x.strip().title(), "%b").month)



education_order = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}
data["education"] = data["education"].map(education_order)
data['job'] = data['job'].str.replace('.', '', regex=False)
data = pd.get_dummies(data, columns=['marital', "contact", "poutcome", "job"], dtype="int")
data["default"] = data["default"].map({"no": 0, "yes": 1})
data["housing"] = data["housing"].map({"no": 0, "yes": 1})
data["loan"] = data["loan"].map({"no": 0, "yes": 1})
data["month"] = data["month"].apply(lambda x: datetime.strptime(x.strip().title(), "%b").month)

x = data.drop(columns=["y"])
y = data["y"]

# ================== 2️⃣ Models ==================
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# ================== 3️⃣ Stratified K-Fold ==================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ================== 4️⃣ Evaluation Function ==================
def evaluate_model(model, X, y, scale=False):
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
    cms = []
    reports = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Standardization if needed
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        # Confusion matrix
        cms.append(confusion_matrix(y_test, y_pred))

        # Classification report as DataFrame
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        reports.append(pd.DataFrame(report).T)  # transpose to have metrics as columns

    # Aggregate metrics
    mean_metrics = {
        "Accuracy": np.mean(accuracy_list),
        "Precision": np.mean(precision_list),
        "Recall": np.mean(recall_list),
        "F1": np.mean(f1_list)
    }

    # Total confusion matrix
    total_cm = np.sum(cms, axis=0)

    # Average classification report
    all_reports = pd.concat(reports)
    avg_report = all_reports.groupby(all_reports.index).mean()

    return mean_metrics, total_cm, avg_report

# ================== 5️⃣ Run Evaluation ==================
for name, model in models.items():
    print(f"\n==============================\nEvaluating {name}...\n")
    scale = True if name == "Logistic Regression" else False
    metrics, cm, avg_report = evaluate_model(model, x, y, scale=scale)

    print("🔹 Mean Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    print("\n📊 Total Confusion Matrix:")
    print(cm)

    print("\n🧾 Average Classification Report:")
    print(avg_report)































import os
import json
import joblib
import shap
import mlflow
import mlflow.keras
from datetime import datetime
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import gc
import xgboost as xgb
import lightgbm as lgb


# ------------------- CONFIG -------------------
N_SPLITS = 10
RANDOM_STATE = 42
EXPERIMENT_NAME = "MultiModel_Training_With_SHAP"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Load Data ----------
train_df = pd.read_csv(r"/content/train.csv")
test_df = pd.read_csv(r"/content/test.csv")

education_order = {"unknown": 0, "primary": 1, "secondary": 2, "tertiary": 3}

def pre_process(data):
    data["education"] = data["education"].map(education_order)
    data['job'] = data['job'].str.replace('.', '', regex=False)
    data = pd.get_dummies(data, columns=['marital', "contact", "poutcome", "job"], dtype="int")
    data["default"] = data["default"].map({"no": 0, "yes": 1})
    data["housing"] = data["housing"].map({"no": 0, "yes": 1})
    data["loan"] = data["loan"].map({"no": 0, "yes": 1})
    data["month"] = data["month"].apply(lambda x: datetime.strptime(x.strip().title(), "%b").month)
    return data

train_df = pre_process(train_df)
test_df = pre_process(test_df)

def clean_numeric_strings(df):
    """Convert strings like '[1.23E-1]' to proper floats, safely."""
    def parse_value(x):
        if isinstance(x, str):
            x = x.strip().replace('[', '').replace(']', '')
            try:
                return float(x)
            except ValueError:
                return np.nan
        return x

    for col in df.columns:
        df[col] = df[col].apply(parse_value)
    df = df.fillna(0)
    return df

# train_df = clean_numeric_strings(train_df)
# test_df = clean_numeric_strings(test_df)


TARGET_COL = "y"

X_train_df = train_df.drop(columns=[TARGET_COL])
y_train = train_df[TARGET_COL].values

if TARGET_COL in test_df.columns:
    X_test_df = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values
    TEST_HAS_LABELS = True
else:
    X_test_df = test_df
    y_test = None
    TEST_HAS_LABELS = False

print(f"Train shape: {X_train_df.shape}, Test shape: {X_test_df.shape}")

# ---------- DNN Builder ----------
def build_dnn(input_dim, hidden_layers=[64, 32, 16]):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def make_dnn_models():
    return {
        "dnn_3layer": lambda: build_dnn(input_dim=X_train_df.shape[1], hidden_layers=[64, 32, 16]),
    }

# ---------- Model Dictionary ----------
models = {
    # "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    # "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    # "xgboost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
    # "lightgbm": lgb.LGBMClassifier(random_state=RANDOM_STATE),
    # # # "svm": SVC(probability=True, random_state=RANDOM_STATE),
    # "naive_bayes": GaussianNB()
}
# models.update(make_dnn_models())

needs_scaling = {"logistic_regression", "svm", "dnn_3layer", "dnn_5layer"}
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ---------- Training Loop ----------
for name, model in models.items():
    print(f"\n🚀 Training model: {name}")
    mlflow.set_experiment(f"{name}_experiment")

    with mlflow.start_run(run_name=f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        acc_list, prec_list, rec_list, f1_list, roc_list = [], [], [], [], []
        cm_sum = np.zeros((2, 2), dtype=int)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_df, y_train), 1):
            X_tr, X_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            if name in needs_scaling:
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_tr_scaled, X_val_scaled = X_tr.values, X_val.values

            if name.startswith("dnn_"):
                keras_model = model()
                keras_model.fit(X_tr_scaled, y_tr, epochs=50, batch_size=8, verbose=0)
                y_prob = keras_model.predict(X_val_scaled).ravel()
                y_pred = (y_prob > 0.5).astype(int)
            else:
                model.fit(X_tr_scaled, y_tr)
                y_pred = model.predict(X_val_scaled)
                try:
                    y_prob = model.predict_proba(X_val_scaled)[:, 1]
                except:
                    y_prob = y_pred

            acc_list.append(accuracy_score(y_val, y_pred))
            prec_list.append(precision_score(y_val, y_pred, zero_division=0))
            rec_list.append(recall_score(y_val, y_pred, zero_division=0))
            f1_list.append(f1_score(y_val, y_pred, zero_division=0))
            try:
                roc_list.append(roc_auc_score(y_val, y_prob))
            except:
                roc_list.append(np.nan)
            cm_sum += confusion_matrix(y_val, y_pred)

        # Final training on full data
        if name in needs_scaling:
            final_scaler = StandardScaler()
            X_train_scaled = final_scaler.fit_transform(X_train_df)
            X_test_scaled = final_scaler.transform(X_test_df)
        else:
            X_train_scaled, X_test_scaled = X_train_df.values, X_test_df.values
            final_scaler = None

        if name.startswith("dnn_"):
            final_model = model()
            final_model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, verbose=0)
            model_path = os.path.join(OUT_DIR, f"{name}.pkl")
            joblib.dump(final_model, model_path)
            input_example = X_train_scaled[:1]
            pred_example = final_model.predict(input_example)
            signature = infer_signature(input_example, pred_example)
            mlflow.keras.log_model(final_model, name=name, input_example=input_example, signature=signature)
        else:
            model.fit(X_train_scaled, y_train)
            model_path = os.path.join(OUT_DIR, f"{name}.pkl")
            joblib.dump(model, model_path)
            input_example = X_train_scaled[:1]
            pred_example = model.predict(input_example)
            signature = infer_signature(input_example, pred_example)
            mlflow.sklearn.log_model(model, name=name, input_example=input_example, signature=signature)

        if final_scaler:
            scaler_path = os.path.join(OUT_DIR, f"{name}_scaler.pkl")
            joblib.dump(final_scaler, scaler_path)
            mlflow.log_artifact(scaler_path)

        # Metrics
        metrics = {
            "Accuracy": np.mean(acc_list),
            "Precision": np.mean(prec_list),
            "Recall": np.mean(rec_list),
            "F1-Score": np.mean(f1_list),
            "ROC-AUC": np.nanmean(roc_list),
        }
        mlflow.log_metrics(metrics)

        # Confusion Matrix
        cm_file = os.path.join(OUT_DIR, f"{name}_confusion_matrix.npy")
        np.save(cm_file, cm_sum)
        mlflow.log_artifact(cm_file)

        # ---------- SHAP Explainability ----------
        print(f"Computing SHAP values for {name}...")

        if name.lower() in ["xgboost"]:
          continue

        # Use DataFrame with feature names
        X_sample = pd.DataFrame(X_train_scaled[:200], columns=X_train_df.columns)
        X_sample = X_sample.apply(lambda col: pd.to_numeric(col, errors='coerce'))
        X_sample = X_sample.fillna(0).astype(np.float32)
        background = X_sample.sample(min(100, len(X_sample)), random_state=42)
        if name.startswith("dnn_"):
            explainer = shap.DeepExplainer(final_model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values
        elif any(x in name.lower() for x in ["randomforest", "xgboost", "lightgbm"]):
            # if name.lower() in "xgboost":
            #   model = XGBWrapper(model)
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        # Save SHAP values
        shap_file = os.path.join(OUT_DIR, f"{name}_shap_values.pkl")
        joblib.dump(shap_values, shap_file)
        mlflow.log_artifact(shap_file)

        # SHAP Summary Plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        shap_summary_path = os.path.join(OUT_DIR, f"{name}_shap_summary.png")
        plt.savefig(shap_summary_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_summary_path)

        # SHAP Bar Plot (Top Features)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        shap_bar_path = os.path.join(OUT_DIR, f"{name}_shap_bar.png")
        plt.savefig(shap_bar_path, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_bar_path)

        # Memory cleanup
        del explainer, shap_values, X_sample
        gc.collect()

        # Test Evaluation
        if TEST_HAS_LABELS:
            if name.startswith("dnn_"):
                y_test_prob = final_model.predict(X_test_scaled).ravel()
                y_test_pred = (y_test_prob > 0.5).astype(int)
            else:
                y_test_pred = model.predict(X_test_scaled)
                try:
                    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]
                except:
                    y_test_prob = y_test_pred

            test_metrics = {
                "Test_Accuracy": accuracy_score(y_test, y_test_pred),
                "Test_F1": f1_score(y_test, y_test_pred),
                "Test_ROC_AUC": roc_auc_score(y_test, y_test_prob)
            }
            mlflow.log_metrics(test_metrics)

        print(f"✅ {name} completed and logged to MLflow.\n")

print("\nAll models trained and logged successfully.")


