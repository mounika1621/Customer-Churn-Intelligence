import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. LOAD DATA
# ==========================================

print("\n===== LOADING DATA =====")

df = pd.read_csv("data/cleaned_customer_churn.csv")

X = df.drop(columns=["customerID", "Churn"])

print("Dataset loaded successfully!")


# ==========================================
# 2. LOAD PREPROCESSOR AND MODEL
# ==========================================

print("\n===== LOADING RANDOM FOREST MODEL =====")

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/random_forest.pkl")

print("Model and preprocessor loaded successfully!")


# ==========================================
# 3. TRANSFORM DATA
# ==========================================

print("\n===== TRANSFORMING DATA =====")

X_processed = preprocessor.transform(X)

print("Data transformation completed!")


# ==========================================
# 4. EXTRACT FEATURE IMPORTANCE
# ==========================================

print("\n===== CALCULATING FEATURE IMPORTANCE =====")

importance = model.feature_importances_

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== TOP 15 IMPORTANT FEATURES =====")

print(importance_df.head(15).to_string(index=False))


# ==========================================
# 5. SAVE FEATURE IMPORTANCE
# ==========================================

importance_df.to_csv(
    "data/feature_importance.csv",
    index=False
)

print("\nFeature importance data saved successfully!")


# ==========================================
# 6. VISUALIZE TOP 15 FEATURES
# ==========================================

top_features = importance_df.head(15).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Customer Churn Features")
plt.xlabel("Feature Importance")
plt.ylabel("Customer Features")

plt.tight_layout()

plt.savefig(
    "screenshots/feature_importance.png",
    dpi=300
)

plt.close()

print("Feature importance graph saved successfully!")


# ==========================================
# 7. COMPLETION
# ==========================================

print("\n===== FEATURE IMPORTANCE ANALYSIS COMPLETED =====")