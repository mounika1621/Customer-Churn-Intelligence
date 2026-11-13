import pandas as pd
import numpy as np

# 1. Load original dataset
file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("Original dataset shape:", df.shape)

# 2. Convert blank spaces into missing values
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

# 3. Convert TotalCharges into numerical format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 4. Handle missing TotalCharges
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 5. Remove duplicate records if any
df = df.drop_duplicates()

# 6. Check missing values
print("\n===== MISSING VALUES AFTER CLEANING =====")
print(df.isnull().sum().sum())

# 7. Check duplicate records
print("\n===== DUPLICATE RECORDS =====")
print(df.duplicated().sum())

# 8. Check data types
print("\n===== UPDATED DATA TYPES =====")
print(df.dtypes)

# 9. Save cleaned dataset
output_path = "data/cleaned_customer_churn.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Final dataset shape:", df.shape)
print("Saved file:", output_path)