import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parent

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ModuleNotFoundError:
    HAS_PLOTTING = False


# Load dataset
df = pd.read_csv(BASE_DIR / "Healthcare_FeatureEngineered.csv")
print("Dataset Loaded:", df.shape)


# Create Risk_Level if missing
if "Risk_Level" not in df.columns:
    def risk_category(row):
        if pd.isna(row["Age"]) or pd.isna(row["Symptom_Count"]):
            return pd.NA
        score = 0
        if row["Age"] > 60:
            score += 2
        elif row["Age"] > 40:
            score += 1

        if row["Symptom_Count"] >= 6:
            score += 2
        elif row["Symptom_Count"] >= 3:
            score += 1

        if score <= 2:
            return "Low"
        if score <= 4:
            return "Medium"
        return "High"

    df["Risk_Level"] = df.apply(risk_category, axis=1)


# Features & target
feature_cols_df = df.drop(columns=["Risk_Level", "Disease", "Patient_ID"], errors="ignore")
required_cols = list(feature_cols_df.columns) + ["Risk_Level"]
df_model = df.dropna(subset=required_cols).copy()
rows_dropped = len(df) - len(df_model)
print(f"Rows dropped due to missing values (no imputation): {rows_dropped}")

y = df_model["Risk_Level"]
X = df_model.drop(columns=["Risk_Level", "Disease", "Patient_ID"], errors="ignore")


# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Identify column types
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
num_cols = X.select_dtypes(include=["number"]).columns


# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded,
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# Decision Tree only
print("\n==============================")
print("Training Decision Tree...")

decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]
)

decision_tree_pipeline.fit(X_train, y_train)
y_pred = decision_tree_pipeline.predict(X_test)
y_prob = decision_tree_pipeline.predict_proba(X_test)[:, 1]


# Metrics
test_acc = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, decision_tree_pipeline.predict(X_train))
test_acc_percent = test_acc * 100
train_acc_percent = train_acc * 100
train_loss_percent = (1 - train_acc) * 100
test_loss_percent = (1 - test_acc) * 100

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"Decision Tree Accuracy Formula: {(y_test == y_pred).sum()} / {len(y_test)}")
print(f"Decision Tree Train Accuracy: {train_acc_percent:.2f}%")
print(f"Decision Tree Test Accuracy: {test_acc_percent:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)


joblib.dump(decision_tree_pipeline, BASE_DIR / "decision_tree_model.pkl")
print("Saved: decision_tree_model.pkl")
print("==============================")


# Graphical representation (Decision Tree only)
if HAS_PLOTTING:
    # Accuracy vs loss graph
    plt.figure(figsize=(8, 5))
    phases = ["Train", "Test"]
    accuracy_values = [train_acc_percent, test_acc_percent]
    loss_values = [train_loss_percent, test_loss_percent]

    plt.plot(phases, accuracy_values, marker="o", linewidth=2.4, label="Accuracy (%)")
    plt.plot(phases, loss_values, marker="o", linewidth=2.4, label="Loss (%)")
    for i, value in enumerate(accuracy_values):
        plt.text(i, value + 0.4, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(loss_values):
        plt.text(i, value + 0.4, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.title("Decision Tree: Accuracy vs Loss")
    plt.xlabel("Dataset Split")
    plt.ylabel("Percentage")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Decision tree metrics graph
    plt.figure(figsize=(9, 5))
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    metric_values = [
        test_acc_percent,
        precision * 100,
        recall * 100,
        f1 * 100,
        roc_auc * 100,
    ]
    plt.plot(metric_names, metric_values, marker="o", linewidth=2.6, color="#0b7a75")
    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.35, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.title("Decision Tree Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Score (%)")
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()

    # Confusion matrix graph
    class_names = label_encoder.classes_
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Decision Tree")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
else:
    print(
        "Plotting skipped: install packages with "
        "\"python -m pip install matplotlib seaborn\""
    )
