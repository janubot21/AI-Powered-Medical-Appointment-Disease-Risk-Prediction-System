import pandas as pd  
import numpy as np  
import re     
import warnings
warnings.filterwarnings("ignore")
# Load the dataset

file_path = "Healthcare.csv"   
df = pd.read_csv(file_path)

print(" Original shape:", df.shape)
print("\n Original dtypes:\n", df.dtypes)

# Basic cleaning (keep text columns as TEXT/strings)
if "Patient_ID" in df.columns:
    df["Patient_ID"] = df["Patient_ID"].astype(str).str.strip()
    
    if df["Patient_ID"].str.fullmatch(r"\d+").all():
        df["Patient_ID"] = df["Patient_ID"].astype("Int64")

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")

if "Symptoms" in df.columns:
    df["Symptoms"] = df["Symptoms"].astype(str).str.strip()

if "Disease" in df.columns:
    df["Disease"] = df["Disease"].astype(str).str.strip()

#before encoding 
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip()

df = df.drop_duplicates()

# Fill missing for TEXT columns
for col in df.select_dtypes(include=["object"]).columns:
    if df[col].isna().any():
        mode_val = df[col].mode(dropna=True)
        fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
        df[col] = df[col].fillna(fill_val)

# Fill missing for NUMERIC columns 
numeric_cols = df.select_dtypes(include=["number", "Int64", "float64", "int64"]).columns
for col in numeric_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    gender_map = {"male": 1, "female": 0, "m": 1, "f": 0}
    df["Gender"] = df["Gender"].map(gender_map).fillna(-1).astype("Int64")


# Symptom_Count (only if needed / correct it)
if "Symptoms" in df.columns:
    def count_symptoms(x: str) -> int:
        x = str(x).strip()
        if x == "" or x.lower() == "nan":
            return 0
        parts = [p.strip() for p in re.split(r"[,;|]+", x) if p.strip()]
        return len(parts)

    df["Symptom_Count"] = df["Symptoms"].apply(count_symptoms).astype("Int64")

# Show preprocessing output (like you wanted)

print("\n Cleaned shape:", df.shape)

print("\n Missing values after preprocessing (output style like you asked):")
missing_report = df.isna().sum().sort_values(ascending=True)
for col, cnt in missing_report.items():
    print(f"{col:<15} {cnt}")

print("\n Final dtypes:\n", df.dtypes)

print("\n Sample output (first 5 rows):")
print(df.head())

# Save cleaned dataset

out_path = "Healthcare_Cleaned.csv"
df.to_csv(out_path, index=False)
print(f"\n Saved cleaned file as: {out_path}")
