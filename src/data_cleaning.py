import json
import os
import numpy as np
import pandas as pd


# =========================================================
# DATA CLEANING
# =========================================================

class DataCleaning:
    def __init__(self, file_path, config_path):
        self.file_path = file_path
        self.config_file = config_path
        self.df = None
    
    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config = json.load(f)
        
        return self.config

    # -----------------------------------------------------
    # Read CSV
    # -----------------------------------------------------

    def read_file(self):
        self.df = pd.read_csv(self.file_path)
        return self.df

    # -----------------------------------------------------
    # Clean String Values
    # Example:
    # admin. -> admin
    # blue-collar -> blue_collar
    # -----------------------------------------------------

    def clean_categorical_values(self):
        categorical_cols = self.df.select_dtypes(include='object').columns

        for col in categorical_cols:

            self.df[col] = (
                                self.df[col]
                                .astype(str)
                                .str.strip()
                                .str.lower()
                                .str.replace('.', '', regex=False)
                                .str.replace('-', '_', regex=False)
                            )
        return self.df

    # -----------------------------------------------------
    # Drop Columns
    # -----------------------------------------------------

    def drop_columns(self):
        existing_cols = [
                            col for col in self.config.get("drop_cols", [])
                            if col in self.df.columns
                        ]
        self.df = self.df.drop(columns=existing_cols)
        return self.df

    # -----------------------------------------------------
    # Convert Numeric Columns
    # -----------------------------------------------------

    def convert_numeric_columns(self):
        numeric_columns = self.config.get("numeric_columns", [])
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        return self.df

    # -----------------------------------------------------
    # Complete Cleaning Pipeline
    # -----------------------------------------------------

    def process(self):
        self.read_file()
        self.load_config()
        self.clean_categorical_values()
        self.drop_columns()
        self.convert_numeric_columns()
        return self.df, self.config


# =========================================================
# PREPROCESSING
# =========================================================

class Preprocessing:
    def __init__(self, data, config, save_data=True, output_path='cleaned_data.csv'):
        self.df = data
        self.config = config
        self.save_data = save_data
        self.output_path = output_path

    # -----------------------------------------------------
    # Binary Encoding
    # yes/no -> 1/0
    # -----------------------------------------------------

    def binary_encoding(self):
        for col in self.config.get("binary_columns"):
            self.df[col] = self.df[col].map({'yes': 1, 'no': 0})
        # return df

    # -----------------------------------------------------
    # Ordinal Encoding
    # -----------------------------------------------------

    def ordinal_encoding(self):
        for col, mapping in self.config.get("ordinal_mappings", {}).items():
            self.df[col] = self.df[col].map(mapping)
        # return self.df

    # -----------------------------------------------------
    # Month Encoding
    # Temporal ordered feature
    # -----------------------------------------------------

    def month_encoding(self):
        self.df['month'] = self.df['month'].map(self.config.get('month_mapping'))
        # return self.df

    # -----------------------------------------------------
    # One Hot Encoding
    # -----------------------------------------------------

    def one_hot_encoding(self):
        self.df = pd.get_dummies(
                                    self.df,
                                    columns=self.config.get('nominal_columns'),
                                    drop_first=False
                                )
        # return df

    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------

    def feature_engineering(self):

        # ---------------------------------------------
        # Prior Contacted
        # ---------------------------------------------

        self.df['prior_contacted'] = (self.df['pdays'] != -1).astype(int)

        # ---------------------------------------------
        # Previous Success
        # ---------------------------------------------

        self.df['previous_success_flag'] = (self.df['poutcome_success'] == 1).astype(int)

        # ---------------------------------------------
        # Debt Burden
        # ---------------------------------------------

        self.df['debt_burden'] = (self.df['housing'] + self.df['loan'])

        # ---------------------------------------------
        # Has Any Loan
        # ---------------------------------------------

        self.df['has_any_loan'] = ((self.df['housing'] == 1) | (self.df['loan'] == 1)).astype(int)

        # ---------------------------------------------
        # Balance To Age
        # ---------------------------------------------

        self.df['balance_to_age'] = (self.df['balance'] / (self.df['age'] + 1))

        # ---------------------------------------------
        # Customer Engagement
        # ---------------------------------------------

        self.df['customer_engagement'] = (self.df['duration'] * self.df['previous'])

        # ---------------------------------------------
        # Log Balance
        # ---------------------------------------------

        # self.df['log_balance'] = np.log1p(np.abs(self.df['balance']))

        # return df

    def save_cleaned_data(self):
        if self.save_data:
            self.df.to_csv(self.output_path, index=False)
            print(f"Cleaned data saved to: {self.output_path}")

    # -----------------------------------------------------
    # Complete Preprocessing
    # -----------------------------------------------------

    def process(self):
        self.binary_encoding()
        self.ordinal_encoding()
        self.month_encoding()
        self.one_hot_encoding()
        self.feature_engineering()
        self.save_cleaned_data()

for csv_file in ["train.csv", "test.csv"]:
    data_cleaner = DataCleaning(
                                    file_path=f"datasets/{csv_file}",
                                    config_path="utils/config.json"
                                )

    cleaned_data, config_data = data_cleaner.process()

    data_preprocessor = Preprocessing(
                                        data=cleaned_data, 
                                        config=config_data, 
                                        output_path=f"datasets/cleaned_{csv_file}"
                                    )
    data_preprocessor.process()