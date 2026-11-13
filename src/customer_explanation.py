import pandas as pd
import joblib
import shap
import numpy as np


# ==========================================
# 1. LOAD MODEL AND PREPROCESSOR
# ==========================================

print("\n===== LOADING MODEL =====")

model = joblib.load("models/random_forest.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

print("Model loaded successfully!")


# ==========================================
# 2. LOAD CUSTOMER DATA
# ==========================================

df = pd.read_csv("data/cleaned_customer_churn.csv")

X = df.drop(columns=["customerID", "Churn"])

# Select one customer for demonstration
customer = X.iloc[[0]]

print("\n===== CUSTOMER DETAILS =====")

print(customer.to_string(index=False))


# ==========================================
# 3. PREDICT CHURN PROBABILITY
# ==========================================

print("\n===== CHURN PREDICTION =====")

customer_processed = preprocessor.transform(customer)

prediction = model.predict(customer_processed)[0]

probability = model.predict_proba(customer_processed)[0][1]

print("Churn probability:", round(probability * 100, 2), "%")

if probability < 0.30:
    risk_level = "LOW"
elif probability < 0.60:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

print("Customer risk level:", risk_level)

if prediction == 1:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")


# ==========================================
# 4. CREATE SHAP EXPLAINER
# ==========================================

print("\n===== GENERATING CUSTOMER EXPLANATION =====")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(customer_processed)

if isinstance(shap_values, list):
    customer_shap = shap_values[1][0]

elif shap_values.ndim == 3:
    customer_shap = shap_values[0, :, 1]

else:
    customer_shap = shap_values[0]


# ==========================================
# 5. IDENTIFY IMPORTANT FEATURES
# ==========================================

feature_names = preprocessor.get_feature_names_out()

explanation_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Value": customer_shap
})

explanation_df["Absolute_Impact"] = (
    explanation_df["SHAP_Value"].abs()
)

explanation_df = explanation_df.sort_values(
    by="Absolute_Impact",
    ascending=False
)

print("\n===== TOP 10 CUSTOMER-SPECIFIC FEATURES =====")

print(
    explanation_df.head(10)[
        ["Feature", "SHAP_Value"]
    ].to_string(index=False)
)


# ==========================================
# 6. POSITIVE AND NEGATIVE CONTRIBUTIONS
# ==========================================

print("\n===== FEATURES INCREASING CHURN SCORE =====")

increasing = explanation_df[
    explanation_df["SHAP_Value"] > 0
]

print(
    increasing.head(5)[
        ["Feature", "SHAP_Value"]
    ].to_string(index=False)
)

print("\n===== FEATURES DECREASING CHURN SCORE =====")

decreasing = explanation_df[
    explanation_df["SHAP_Value"] < 0
]

print(
    decreasing.head(5)[
        ["Feature", "SHAP_Value"]
    ].to_string(index=False)
)


# ==========================================
# 7. RETENTION RECOMMENDATIONS
# ==========================================

print("\n===== RETENTION RECOMMENDATIONS =====")

recommendations = []

contract = customer["Contract"].iloc[0]
tenure = customer["tenure"].iloc[0]
security = customer["OnlineSecurity"].iloc[0]
support = customer["TechSupport"].iloc[0]
monthly_charges = customer["MonthlyCharges"].iloc[0]

if contract == "Month-to-month":
    recommendations.append(
        "Offer a suitable long-term contract with clear benefits."
    )

if tenure < 12:
    recommendations.append(
        "Provide onboarding support and early customer engagement."
    )

if security == "No":
    recommendations.append(
        "Offer online security services or a suitable package."
    )

if support == "No":
    recommendations.append(
        "Introduce technical support benefits."
    )

if monthly_charges > 70:
    recommendations.append(
        "Review the customer's billing plan and offer suitable pricing."
    )

if risk_level == "HIGH":
    recommendations.append(
        "Prioritize this customer for a retention follow-up."
    )

if not recommendations:
    recommendations.append(
        "Continue regular engagement and monitor customer satisfaction."
    )

for number, recommendation in enumerate(recommendations, start=1):
    print(f"{number}. {recommendation}")


# ==========================================
# 8. SAVE CUSTOMER EXPLANATION
# ==========================================

explanation_df.to_csv(
    "data/customer_shap_explanation.csv",
    index=False
)

print("\nCustomer explanation saved successfully!")

print("\n===== CUSTOMER EXPLANATION COMPLETED =====")