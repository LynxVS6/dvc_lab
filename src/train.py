import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

Path("models").mkdir(exist_ok=True)

df = pd.read_csv("data/features.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])

params = {
    "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    "model__fit_intercept": [True, False],
}

grid = GridSearchCV(pipe, params, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

joblib.dump(grid.best_estimator_, "models/model.joblib")

X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("model saved to models/model.joblib")
