import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# =====================================
# 1. LOAD PROCESSED DATA
# =====================================

print("===== LOADING PROCESSED DATA =====")

data = np.load("data/model_data.npz")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# =====================================
# 2. CREATE LOGISTIC REGRESSION MODEL
# =====================================

print("\n===== TRAINING LOGISTIC REGRESSION =====")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Logistic Regression training completed!")


# =====================================
# 3. MAKE PREDICTIONS
# =====================================

print("\n===== MAKING PREDICTIONS =====")

y_pred = model.predict(X_test)

# Predict probability of churn
y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


# =====================================
# 4. MODEL EVALUATION
# =====================================

print("\n===== MODEL EVALUATION =====")

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")


# =====================================
# 5. CLASSIFICATION REPORT
# =====================================

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"]
    )
)


# =====================================
# 6. CONFUSION MATRIX
# =====================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "screenshots/logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix graph saved successfully!")


# =====================================
# 7. SAVE TRAINED MODEL
# =====================================

print("\n===== SAVING MODEL =====")

joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

print("Logistic Regression model saved successfully!")


# =====================================
# 8. FINAL STATUS
# =====================================

print("\n===== LOGISTIC REGRESSION COMPLETED =====")