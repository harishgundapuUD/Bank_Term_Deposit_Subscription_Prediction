
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

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Encoding categorical variables
6. Train/Test Split
7. Model Training
8. Model Evaluation

## 🤖 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- (Optional: XGBoost, Gradient Boosting)

## 📈 Evaluation Metrics

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- Confusion Matrix
- ROC-AUC Score

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## 📁 Project Structure

Bank_Term_Deposit_Subscription_Prediction/
├── data/
├── notebooks/
├── src/
├── models/
├── requirements.txt
└── README.md

## ⚙️ Installation

```bash
git clone https://github.com/harishgundapuUD/Bank_Term_Deposit_Subscription_Prediction.git
cd Bank_Term_Deposit_Subscription_Prediction
pip install -r requirements.txt


🚀 Usage

Run training script:

python src/train.py

Or open notebook:

jupyter notebook
📊 Results

The model predicts whether a customer will subscribe to a term deposit with good accuracy, helping improve marketing targeting and conversion rates.

🔮 Future Improvements
Hyperparameter tuning
XGBoost / LightGBM models
Deployment using Flask or FastAPI
Streamlit dashboard
Feature importance using SHAP
👨‍💻 Author

Harish Gundapu

📄 License

This project is open-source under the MIT License.
```
