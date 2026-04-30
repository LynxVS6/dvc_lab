import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

Path("metrics").mkdir(exist_ok=True)

model = joblib.load("models/model.joblib")
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").iloc[:, 0]

pred = model.predict(X_test)

metrics = {
    "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    "mae": float(mean_absolute_error(y_test, pred)),
    "r2": float(r2_score(y_test, pred)),
}

with open("metrics/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(metrics)
