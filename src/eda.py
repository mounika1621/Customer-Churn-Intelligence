import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 1. LOAD CLEANED DATASET
# =====================================

df = pd.read_csv("data/cleaned_customer_churn.csv")

sns.set_theme(style="whitegrid")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =====================================
# 2. CUSTOMER CHURN DISTRIBUTION
# =====================================

churn_counts = df["Churn"].value_counts()

print("\n===== CUSTOMER CHURN DISTRIBUTION =====")
print(churn_counts)

print("\n===== CHURN PERCENTAGE =====")

churn_percentage = (
    df["Churn"].value_counts(normalize=True) * 100
)

print(churn_percentage.round(2))


# Create churn distribution graph
plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution", fontsize=16)
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "screenshots/churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nChurn distribution graph saved successfully!")


# =====================================
# 3. CHURN ANALYSIS BY CONTRACT TYPE
# =====================================

df["Churn_Numeric"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

contract_churn = (
    df.groupby("Contract")["Churn_Numeric"]
    .mean() * 100
)

print("\n===== CHURN RATE BY CONTRACT TYPE =====")

print(contract_churn.round(2))


# Create contract churn graph
plt.figure(figsize=(9, 6))

sns.barplot(
    x=contract_churn.index,
    y=contract_churn.values,
    hue=contract_churn.index,
    palette="viridis",
    legend=False
)

plt.title(
    "Customer Churn Rate by Contract Type",
    fontsize=16
)

plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "screenshots/churn_by_contract.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nContract churn graph saved successfully!")


# =====================================
# 4. FINAL SUMMARY
# =====================================

print("\n===== EDA SUMMARY =====")

print("Total Customers:", len(df))

print(
    "Customers Who Stayed:",
    (df["Churn"] == "No").sum()
)

print(
    "Customers Who Left:",
    (df["Churn"] == "Yes").sum()
)

print("\nAll EDA graphs generated successfully!")


# =====================================
# 5. CUSTOMER TENURE ANALYSIS
# =====================================

print("\n===== CUSTOMER TENURE ANALYSIS =====")

# Calculate average tenure
tenure_analysis = df.groupby("Churn")["tenure"].mean()

print("Average Tenure:")
print(tenure_analysis.round(2))

# Create boxplot
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Tenure vs Churn", fontsize=16)
plt.xlabel("Customer Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    "screenshots/tenure_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nTenure analysis graph saved successfully!")

# =====================================
# 6. MONTHLY CHARGES ANALYSIS
# =====================================

print("\n===== MONTHLY CHARGES ANALYSIS =====")

# Calculate average monthly charges
charges_analysis = df.groupby("Churn")["MonthlyCharges"].mean()

print("Average Monthly Charges:")
print(charges_analysis.round(2))


# Create boxplot
plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Monthly Charges vs Customer Churn", fontsize=16)
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "screenshots/monthly_charges_vs_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nMonthly charges graph saved successfully!")

# =====================================
# 7. PAYMENT METHOD VS CHURN
# =====================================

print("\n===== PAYMENT METHOD VS CHURN =====")

payment_analysis = df.groupby("PaymentMethod")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("Churn Rate by Payment Method:")
print(payment_analysis.round(2))

# Create bar chart
plt.figure(figsize=(10, 6))

sns.barplot(
    x=payment_analysis.index,
    y=payment_analysis.values,
    hue=payment_analysis.index,
    palette="Set2",
    legend=False
)

plt.title("Churn Rate by Payment Method", fontsize=16)
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "screenshots/churn_by_payment_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPayment method analysis graph saved successfully!")

# =====================================
# 8. INTERNET SERVICE VS CHURN
# =====================================

print("\n===== INTERNET SERVICE VS CHURN =====")

internet_analysis = df.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("Churn Rate by Internet Service:")
print(internet_analysis.round(2))

# Create bar chart
plt.figure(figsize=(9, 6))

sns.barplot(
    x=internet_analysis.index,
    y=internet_analysis.values,
    hue=internet_analysis.index,
    palette="Set2",
    legend=False
)

plt.title("Churn Rate by Internet Service", fontsize=16)
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    "screenshots/churn_by_internet_service.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nInternet service analysis graph saved successfully!")

# =====================================
# 9. ONLINE SECURITY VS CHURN
# =====================================

print("\n===== ONLINE SECURITY VS CHURN =====")

security_analysis = df.groupby("OnlineSecurity")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("Churn Rate by Online Security:")
print(security_analysis.round(2))

plt.figure(figsize=(9, 6))

sns.barplot(
    x=security_analysis.index,
    y=security_analysis.values,
    hue=security_analysis.index,
    palette="Set2",
    legend=False
)

plt.title("Churn Rate by Online Security", fontsize=16)
plt.xlabel("Online Security")
plt.ylabel("Churn Rate (%)")

plt.tight_layout()

plt.savefig(
    "screenshots/churn_by_online_security.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nOnline security analysis graph saved successfully!")