# Bank Term Deposit Subscription Prediction

This project is a machine learning classification system that predicts whether a customer will subscribe to a bank term deposit based on demographic, financial, and campaign-related data.

Repository: https://github.com/harishgundapuUD/Bank_Term_Deposit_Subscription_Prediction.git

## 📌 Problem Statement

Banks run marketing campaigns to encourage customers to subscribe to term deposits. However, targeting all customers is inefficient. This project helps identify potential customers who are more likely to subscribe, improving marketing effectiveness and reducing cost.

## 🎯 Objective

To build a machine learning model that predicts whether a customer will subscribe to a term deposit (`yes` or `no`) using historical banking data.

## 📊 Dataset

The dataset contains customer and campaign-related attributes:

- Age
- Job
- Marital status
- Education
- Default status
- Account balance
- Housing loan
- Personal loan
- Contact communication type
- Day and month of last contact
- Campaign-related features (duration, previous outcome)

Target variable: `y` (subscription to term deposit)

## 🧠 Machine Learning Workflow

1. Data Cleaning
2. Feature Engineering
3. Encoding categorical variables
4. Model Training
5. Model Evaluation

## 🤖 Models Used

- Logistic Regression
- Random Forest
- (Advanced: XGBoost, LightGBM)

## 📈 Evaluation Metrics

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- MLFlow

## 🏗️ Project Structure

```bash
BANK_TERM_DEPOSIT_SUBSCRIPTION_PREDICTION/

├── datasets/
│   ├── csv files
│
├── src/
│   ├── data_cleaning.py
│   ├── ensembel_model.py
│   ├── training.py
│
├── trained_models/
│   ├── base_models/
│   │   ├── mlruns/
│   │   └── model_metrics.json
│   │
│   ├── advanced_models/
│   │   ├── mlruns/
│   │   └── model_metrics.json
│   │
│   ├── emsembel_models/
│   │   ├── mlruns/
│   │   └── model_metrics.json
│
├── utils/
│   ├── config.json
│   └── train_columns.json
│
├── app.py
├── requirements.txt
├── README.md
```

⚙️ Installation


```bash
git clone https://github.com/harishgundapuUD/Bank_Term_Deposit_Subscription_Prediction.git
cd Bank_Term_Deposit_Subscription_Prediction
pip install -r requirements.txt
```


🚀 Usage

Run training script:

python src/ensembel_model.py

Or open notebook:

jupyter notebook

📊 Results

The model predicts whether a customer will subscribe to a term deposit with good accuracy, helping improve marketing targeting and conversion rates.

🔮 Future Improvements
Hyperparameter tuning
XGBoost / LightGBM models
Streamlit dashboard

👨‍💻 Author

Harish Gundapu
