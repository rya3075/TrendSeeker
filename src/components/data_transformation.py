import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    transformed_train_path = os.path.join('artifacts', "train_transformed.npy")
    transformed_test_path = os.path.join('artifacts', "test_transformed.npy")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def compute_indicators(self, df):
        """Compute technical indicators and generate target labels."""
        try:
            df = df.copy()

            # Convert necessary columns to numeric
            numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Drop rows with missing values in essential columns
            df.dropna(subset=numeric_cols, inplace=True)

            # Compute Exponential Moving Average (EMA)
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()

            # Compute Stochastic RSI if 'RSI' exists
            if 'RSI' in df.columns:
                min_rsi = df['RSI'].rolling(window=5).min()
                max_rsi = df['RSI'].rolling(window=5).max()
                df['Stoch_RSI_5'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
            else:
                df['Stoch_RSI_5'] = 0  # Default value if RSI is missing

            df['Stoch_RSI_5'].fillna(0, inplace=True)

            # Generate 'Trend' column: 1 if next Close price is higher, else 0
            df['Trend'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def get_preprocessor(self):
        """Create and return a preprocessing pipeline."""
        try:
            return Pipeline([("scaler", StandardScaler())])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """Perform data transformation and save the preprocessor."""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test data successfully")

            # Compute indicators and generate 'Trend'
            train_df = self.compute_indicators(train_df)
            test_df = self.compute_indicators(test_df)

            # Feature selection
            feature_columns = ["Open", "High", "Low", "Close", "Volume", "EMA_5", "Stoch_RSI_5"]
            target_column = "Trend"

            if target_column not in train_df or target_column not in test_df:
                raise CustomException(f"Target column '{target_column}' is missing after processing!", sys)

            # Separate features & target
            X_train, y_train = train_df[feature_columns], train_df[target_column]
            X_test, y_test = test_df[feature_columns], test_df[target_column]

            # Get preprocessing pipeline
            preprocessor = self.get_preprocessor()
            
            logging.info("Applying preprocessing pipeline")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Combine features & target
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Save transformed data
            np.save(self.data_transformation_config.transformed_train_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_path, test_arr)

            # Save preprocessor
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info("Preprocessing object and transformed datasets saved successfully")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)