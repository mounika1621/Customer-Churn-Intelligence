import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# =====================================
# 1. LOAD CLEANED DATASET
# =====================================

print("===== LOADING DATASET =====")

df = pd.read_csv("data/cleaned_customer_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =====================================
# 2. SEPARATE FEATURES AND TARGET
# =====================================

print("\n===== SEPARATING FEATURES AND TARGET =====")

# Remove customerID
X = df.drop(columns=["customerID", "Churn"])

# Convert target into numerical values
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# =====================================
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# =====================================

print("\n===== IDENTIFYING FEATURE TYPES =====")

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_features = X.select_dtypes(
    include=["str"]
).columns.tolist()

print("Numerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# =====================================
# 4. SPLIT TRAINING AND TESTING DATA
# =====================================

print("\n===== SPLITTING DATASET =====")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# =====================================
# 5. FEATURE ENCODING AND SCALING
# =====================================

print("\n===== FEATURE ENCODING =====")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

print("Feature encoding completed!")

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# =====================================
# 6. SAVE PREPROCESSOR AND DATA
# =====================================

print("\n===== SAVING PROCESSED DATA =====")

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

np.savez_compressed(
    "data/model_data.npz",
    X_train=X_train_processed,
    X_test=X_test_processed,
    y_train=y_train.to_numpy(),
    y_test=y_test.to_numpy()
)

print("Preprocessor saved successfully!")
print("Processed dataset saved successfully!")

print("\n===== DATA PREPARATION COMPLETED =====")