import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

print("===== LOADING CLEANED DATA =====")

df = pd.read_csv("data/cleaned_customer_churn.csv")

df = df.drop(columns=["customerID"])

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

X = df.drop(columns=["Churn"])
y = df["Churn"]

print("Dataset shape:", X.shape)


print("\n===== PREPARING TRAINING DATA =====")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["str"]
).columns.tolist()


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


print("\n===== INITIALIZING MODELS =====")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}


print("\n===== CORRECTED STRATIFIED 5-FOLD CROSS-VALIDATION =====")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

results = {}

for model_name, model in models.items():

    print(f"\nEvaluating: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    results[model_name] = {}

    for metric_name in scoring:

        metric_scores = scores[f"test_{metric_name}"]

        mean_score = np.mean(metric_scores)
        std_score = np.std(metric_scores)

        results[model_name][metric_name] = mean_score

        print(
            f"{metric_name.upper():12}: "
            f"{mean_score:.4f} (+/- {std_score:.4f})"
        )


print("\n===== FINAL CROSS-VALIDATION SUMMARY =====")

for model_name, metrics in results.items():

    print(f"\n{model_name}")

    for metric_name, score in metrics.items():

        print(f"{metric_name.upper():12}: {score:.4f}")


best_model = max(
    results,
    key=lambda name: results[name]["f1"]
)

print("\n===== BEST MODEL BASED ON MEAN F1 SCORE =====")

print("Best model:", best_model)
print("Mean F1 Score:", round(results[best_model]["f1"], 4))

print("\n===== CORRECTED CROSS-VALIDATION COMPLETED =====")