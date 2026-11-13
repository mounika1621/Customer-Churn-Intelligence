import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("===== LOADING PROCESSED DATA =====")

data = np.load("data/model_data.npz")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


print("\n===== TRAINING DECISION TREE =====")

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("Decision Tree training completed!")


print("\n===== MAKING PREDICTIONS =====")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


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


print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Stayed", "Churned"]
    )
)


print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    "screenshots/decision_tree_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix graph saved successfully!")


print("\n===== SAVING MODEL =====")

joblib.dump(model, "models/decision_tree.pkl")

print("Decision Tree model saved successfully!")

print("\n===== DECISION TREE COMPLETED =====")