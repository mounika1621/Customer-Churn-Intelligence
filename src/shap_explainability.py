import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ==========================================
# 1. LOAD DATASET
# ==========================================

print("\n===== LOADING DATASET =====")

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

feature_names = preprocessor.get_feature_names_out()

# Select 500 customers for SHAP analysis
X_sample = X_processed[:500]

print("Selected customers:", X_sample.shape[0])
print("Total features:", X_sample.shape[1])


# ==========================================
# 4. CREATE SHAP EXPLAINER
# ==========================================

print("\n===== CREATING SHAP EXPLAINER =====")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully!")


# ==========================================
# 5. CALCULATE SHAP VALUES
# ==========================================

print("\n===== CALCULATING SHAP VALUES =====")

shap_values = explainer.shap_values(X_sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    churn_shap_values = shap_values[1]

elif shap_values.ndim == 3:
    churn_shap_values = shap_values[:, :, 1]

else:
    churn_shap_values = shap_values

print("SHAP values calculated successfully!")


# ==========================================
# 6. GLOBAL FEATURE IMPORTANCE
# ==========================================

print("\n===== GLOBAL SHAP FEATURE IMPORTANCE =====")

mean_importance = abs(churn_shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_SHAP_Importance": mean_importance
})

shap_importance = shap_importance.sort_values(
    by="Mean_SHAP_Importance",
    ascending=False
)

print("\nTOP 15 SHAP FEATURES:")

print(shap_importance.head(15).to_string(index=False))

shap_importance.to_csv(
    "data/shap_feature_importance.csv",
    index=False
)


# ==========================================
# 7. SHAP SUMMARY PLOT
# ==========================================

print("\n===== GENERATING SHAP SUMMARY PLOT =====")

shap.summary_plot(
    churn_shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "screenshots/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary plot saved successfully!")


# ==========================================
# 8. SHAP BAR PLOT
# ==========================================

print("\n===== GENERATING SHAP BAR PLOT =====")

shap.summary_plot(
    churn_shap_values,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "screenshots/shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("SHAP bar plot saved successfully!")


# ==========================================
# 9. COMPLETION
# ==========================================

print("\n===== SHAP EXPLAINABILITY COMPLETED =====")