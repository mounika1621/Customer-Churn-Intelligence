import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n===== LOADING DATASET =====")

df = pd.read_csv("data/cleaned_customer_churn.csv")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

print("Dataset loaded successfully!")


# ==========================================
# 2. SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Test samples:", len(X_test))


# ==========================================
# 3. LOAD TUNED MODEL
# ==========================================

print("\n===== LOADING TUNED RANDOM FOREST =====")

model = joblib.load(
    "models/tuned_random_forest_pipeline.pkl"
)

print("Tuned model loaded successfully!")


# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

print("\n===== MAKING PREDICTIONS =====")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions completed!")


# ==========================================
# 5. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n===== TUNED RANDOM FOREST RESULTS =====")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"]
    )
)


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title("Tuned Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "screenshots/tuned_random_forest_confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix graph saved successfully!")


# ==========================================
# 8. COMPLETION
# ==========================================

print("\n===== TUNED MODEL EVALUATION COMPLETED =====")