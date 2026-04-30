from sklearn.datasets import load_diabetes
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

data = load_diabetes(as_frame=True)
df = data.frame.copy()

df = df.dropna().reset_index(drop=True)

df.to_csv("data/clean.csv", index=False)

print("clean dataset saved to data/clean.csv")
