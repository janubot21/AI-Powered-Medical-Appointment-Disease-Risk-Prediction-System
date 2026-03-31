import pandas as pd
import numpy as np
import re
import warnings
from pathlib import Path
warnings.filterwarnings("ignore") 

# Load CLEANED dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "Healthcare_Cleaned.csv")
print("Cleaned dataset loaded")
print("Shape:", df.shape)

#  Create Symptom List Column
def split_symptoms(text):
    if pd.isna(text):
        return pd.NA
    text = str(text).lower().strip()
    if text in ["", "none"]:
        return pd.NA
    return [s.strip() for s in re.split(r"[,;|]+", text) if s.strip()]

df["Symptoms_List"] = df["Symptoms"].apply(split_symptoms)

#  Symptom Count Feature
df["Symptom_Count"] = df["Symptoms_List"].apply(
    lambda x: len(x) if isinstance(x, list) else pd.NA
).astype("Int64")

# Create Binary Symptom Columns
# Get top 15 most common symptoms
all_symptoms = [
    sym for sublist in df["Symptoms_List"] if isinstance(sublist, list) for sym in sublist
]
top_symptoms = pd.Series(all_symptoms).value_counts().head(15).index

for symptom in top_symptoms:
    safe = re.sub(r"[^a-z0-9_]+", "_", symptom.replace(" ", "_"))
    df[f"SYM_{safe}"] = df["Symptoms_List"].apply(
        lambda x: (1 if symptom in x else 0) if isinstance(x, list) else pd.NA
    ).astype("Int64")

print(" Symptom binary features created")
#  Age Group Feature
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 18, 35, 50, 65, 100],
    labels=["Child", "Young", "Adult", "Middle_Age", "Senior"]
)

#  BMI Category (if BMI exists)
if "BMI" in df.columns:
    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

#  Blood Pressure Category
if "BloodPressure" in df.columns:
    df["BP_Category"] = pd.cut(
        df["BloodPressure"],
        bins=[0, 80, 120, 140, 300],
        labels=["Low", "Normal", "Prehypertension", "High"]
    )
# Drop Temporary Column
df.drop(columns=["Symptoms_List"], inplace=True)
# Save Feature Engineered Dataset
df.to_csv(BASE_DIR / "Healthcare_FeatureEngineered.csv", index=False)
print(" Feature engineered dataset saved as 'Healthcare_FeatureEngineered.csv'")
print("\n Sample Columns:")
print(df.columns[:25])
print("Feature Engineering done")
