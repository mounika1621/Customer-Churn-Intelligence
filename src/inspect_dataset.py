import csv
from collections import Counter

file_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    data = list(reader)

# 1. Dataset dimensions
print("===== DATASET INFORMATION =====")
print("Total Rows:", len(data))
print("Total Columns:", len(data[0]))

# 2. Column names
print("\n===== COLUMN NAMES =====")

for column in data[0].keys():
    print(column)

# 3. Missing values
print("\n===== MISSING VALUES =====")

for column in data[0].keys():
    missing = sum(
        1 for row in data
        if row[column].strip() == ""
    )
    print(column, ":", missing)

# 4. Duplicate records
print("\n===== DUPLICATE RECORDS =====")

unique_rows = set(
    tuple(row.values()) for row in data
)

print("Duplicate Rows:", len(data) - len(unique_rows))

# 5. Churn distribution
print("\n===== CHURN DISTRIBUTION =====")

churn_counts = Counter(row["Churn"] for row in data)

for category, count in churn_counts.items():
    percentage = (count / len(data)) * 100

    print(
        category,
        ":",
        count,
        f"({percentage:.2f}%)"
    )