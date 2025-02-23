import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, classification_report
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting data into features and target variable")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "RandomForest": RandomForestClassifier(),
                "GradientBoosting": GradientBoostingClassifier(),
                "SVC": SVC()
            }

            params = {
                "RandomForest": {"n_estimators": [50, 100, 150]},
                "GradientBoosting": {"learning_rate": [0.01, 0.1, 0.2]},
                "SVC": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
            }

            logging.info("Evaluating models using recall as the primary metric")
            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
            
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model selected based on recall: {best_model_name}")
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            recall = recall_score(y_test, y_pred, pos_label=1)

            logging.info(f"Best Model Recall: {recall}")
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info("Model saved successfully")

            return recall
        except Exception as e:
            raise CustomException(e, sys)