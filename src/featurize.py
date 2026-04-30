import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

df = pd.read_csv("data/clean.csv")

df["bmi_bp"] = df["bmi"] * df["bp"]
df["s1_s2_diff"] = df["s1"] - df["s2"]
df["age2"] = df["age"] ** 2

df.to_csv("data/features.csv", index=False)

print("features dataset saved to data/features.csv")
