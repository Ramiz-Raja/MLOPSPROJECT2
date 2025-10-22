# src/utils.py
import os
import joblib
from sklearn.datasets import load_iris
import pandas as pd

def load_dataset():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    df = pd.concat([X, y.rename("target")], axis=1)
    return df

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
