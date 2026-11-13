import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n===== LOADING DATASET =====")

df = pd.read_csv("data/cleaned_customer_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. PREPARE FEATURES AND TARGET
# ==========================================

print("\n===== PREPARING DATA =====")

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["str"]).columns.tolist()

print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. CREATE PREPROCESSING PIPELINE
# ==========================================

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


# ==========================================
# 5. CREATE RANDOM FOREST PIPELINE
# ==========================================

print("\n===== CREATING RANDOM FOREST PIPELINE =====")

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ==========================================
# 6. DEFINE HYPERPARAMETER GRID
# ==========================================

param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [8, 12],
    "classifier__min_samples_split": [10, 20],
    "classifier__min_samples_leaf": [5, 10]
}


# ==========================================
# 7. GRID SEARCH CROSS VALIDATION
# ==========================================

print("\n===== STARTING GRID SEARCH =====")

cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=1,
    verbose=2
)

grid_search.fit(X_train, y_train)


# ==========================================
# 8. DISPLAY BEST RESULTS
# ==========================================

print("\n===== BEST HYPERPARAMETERS =====")

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(round(grid_search.best_score_ * 100, 2), "%")


# ==========================================
# 9. SAVE TUNED MODEL
# ==========================================

print("\n===== SAVING TUNED MODEL =====")

joblib.dump(
    grid_search.best_estimator_,
    "models/tuned_random_forest_pipeline.pkl"
)

print("Tuned Random Forest model saved successfully!")

print("\n===== HYPERPARAMETER TUNING COMPLETED =====")