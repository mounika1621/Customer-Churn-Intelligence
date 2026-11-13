# Customer Churn Intelligence

An end-to-end Machine Learning project that analyzes customer behavior, predicts customer churn, explains model predictions, and provides data-driven customer retention recommendations.

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave helps businesses take proactive retention measures.

This project uses the IBM Telco Customer Churn dataset to explore customer behavior, train multiple machine learning models, compare their performance, and provide an interactive dashboard for churn analysis and prediction.

## Key Features

- Customer churn data analysis and visualization.
- Data preprocessing and feature engineering.
- Machine learning model training and evaluation.
- Comparison of Logistic Regression, Decision Tree, and Random Forest.
- Random Forest hyperparameter tuning.
- 5-fold stratified cross-validation.
- Feature importance analysis.
- SHAP-based explainable AI.
- Individual customer churn prediction.
- Customer risk classification.
- Rule-based retention recommendations.
- Interactive Streamlit dashboard.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SHAP
- Streamlit
- Joblib

## Machine Learning Models

Three primary machine learning algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

A tuned Random Forest model was also evaluated using GridSearchCV.

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.21% |
| Decision Tree | 73.95% | 50.60% | 78.61% | 61.57% | 82.95% |
| Random Forest | 76.15% | 53.51% | 77.54% | 63.32% | 84.12% |
| Tuned Random Forest | 80.34% | 67.51% | 50.00% | 57.45% | 84.34% |

**Selected Model: Random Forest**

The original Random Forest model was selected because it achieved the highest holdout F1 score (63.32%) among the evaluated models while maintaining a recall of 77.54%.

Its 5-fold cross-validation F1 score was 62.84%.

## Explainable AI

SHAP (SHapley Additive exPlanations) was used to understand the contribution of individual features to model predictions.

Important churn-related features identified include:

- Month-to-month contracts
- Customer tenure
- Fiber optic internet service
- Online security
- Total charges
- Technical support
- Electronic check payments

These findings represent model associations and do not establish causation.

## Streamlit Dashboard

The interactive dashboard includes six sections:

1. **Dashboard** – Overall customer statistics and churn distribution.
2. **Customer Analysis** – Explore customers using filters and visualizations.
3. **Churn Prediction** – Predict churn probability for an individual customer.
4. **Explainable AI** – Understand individual predictions using SHAP.
5. **Retention Recommendations** – View rule-based customer retention suggestions.
6. **Model Performance** – Compare model metrics and cross-validation results.

## Project Structure

```text
Customer-Churn-Intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── feature_importance.csv
│   ├── shap_feature_importance.csv
│   └── customer_shap_explanation.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── tuned_random_forest_pipeline.pkl
│   └── preprocessor.pkl
│
├── screenshots/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── model comparison and evaluation charts
│
├── src/
│   ├── inspect_dataset.py
│   ├── preprocess_data.py
│   ├── eda.py
│   ├── prepare_model_data.py
│   ├── train_logistic_regression.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── compare_models.py
│   ├── cross_validation.py
│   ├── tune_random_forest.py
│   ├── feature_importance.py
│   ├── shap_explainability.py
│   └── customer_explanation.py
│
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer-Churn-Intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the dataset

Download the IBM Telco Customer Churn dataset and place the original CSV file inside the `data/` folder.

Expected filename:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### 6. Run the dashboard

```bash
streamlit run app/app.py
```

The dashboard will open in your browser.

## Dataset

**Dataset:** IBM Telco Customer Churn

- Total customers: 7,043
- Features: 20 input attributes
- Target variable: Churn
- Churned customers: 1,869
- Retained customers: 5,174

## Important Notes

- The dataset contains customer information intended for machine learning research and analysis.
- The project uses a held-out test set for model evaluation.
- Cross-validation is performed on training data using a pipeline to avoid preprocessing leakage.
- Retention suggestions are rule-based and should not be interpreted as guaranteed outcomes.
- Model predictions are estimates, not certainties.

## Author

Developed as a Machine Learning and Data Analytics portfolio project.