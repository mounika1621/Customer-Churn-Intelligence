import matplotlib.pyplot as plt
import numpy as np

print("===== MODEL COMPARISON =====")

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest"
]

accuracy = [80.55, 73.95, 76.15]
precision = [65.72, 50.60, 53.51]
recall = [55.88, 78.61, 77.54]
f1_score = [60.40, 61.57, 63.32]
roc_auc = [84.21, 82.95, 84.12]

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score,
    "ROC-AUC": roc_auc
}

print("\n===== PERFORMANCE SUMMARY =====")

for model_index in range(len(models)):
    print(f"\n{models[model_index]}")
    for metric_name, values in metrics.items():
        print(f"{metric_name}: {values[model_index]:.2f}%")


print("\n===== GENERATING COMPARISON GRAPH =====")

x = np.arange(len(models))
width = 0.15

plt.figure(figsize=(13, 7))

for index, (metric_name, values) in enumerate(metrics.items()):
    positions = x + (index - 2) * width

    bars = plt.bar(
        positions,
        values,
        width,
        label=metric_name
    )

    plt.bar_label(
        bars,
        fmt="%.1f",
        padding=2,
        fontsize=8
    )

plt.xlabel("Machine Learning Models")
plt.ylabel("Performance (%)")
plt.title("Customer Churn Prediction - Model Comparison")

plt.xticks(x, models)
plt.ylim(0, 110)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    "screenshots/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Model comparison graph saved successfully!")


print("\n===== BEST MODEL BY F1 SCORE =====")

best_index = np.argmax(f1_score)

print("Best model:", models[best_index])
print("F1 Score:", f1_score[best_index], "%")

print("\n===== MODEL COMPARISON COMPLETED =====")