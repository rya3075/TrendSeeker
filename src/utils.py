import os
import sys
import dill
import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import recall_score

def save_object(file_path, obj):
    """
    Save a Python object using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load a Python object using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple classification models using TimeSeriesSplit cross-validation.
    Returns a report with test recall scores.
    """
    try:
        report = {}
        tscv = TimeSeriesSplit(n_splits=3)

        for model_name, model in models.items():
            parameters = param.get(model_name, {})
            
            if parameters:
                gs = GridSearchCV(model, parameters, cv=tscv, scoring='recall')
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            recall = recall_score(y_test, y_test_pred, average="binary")
            report[model_name] = recall

        return report
    except Exception as e:
        raise CustomException(e, sys)