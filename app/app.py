import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #f4f7fb;
}

/* Main Content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main Headings */
h1 {
    color: #172554;
    font-weight: 800;
}

h2, h3 {
    color: #1e3a5f;
    font-weight: 700;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #edf4ff);
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #dbe5f1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Metric Labels */
[data-testid="stMetricLabel"] {
    color: #475569;
    font-weight: 600;
}

/* Metric Values */
[data-testid="stMetricValue"] {
    color: #0f766e;
    font-weight: 800;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #10243e;
}

[data-testid="stSidebar"] * {
    color: white;
}

/* Sidebar Radio */
[data-testid="stSidebar"] label {
    font-weight: 500;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background-color: #0f766e;
    color: white;
    border-radius: 9px;
    border: none;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background-color: #115e59;
    color: white;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #dbe5f1;
    border-radius: 10px;
}

/* Horizontal Divider */
hr {
    border-color: #dbe5f1;
}

/* Information Boxes */
.stAlert {
    border-radius: 10px;
}

/* Chart Containers */
[data-testid="stImage"],
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD DATASET
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_customer_churn.csv")


df = load_data()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🔍 Churn Intelligence")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Customer Analysis",
        "Churn Prediction",
        "Explainable AI",
        "Retention Recommendations",
        "Model Performance"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-powered customer churn prediction "
    "and retention intelligence system."
)


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("📊 Customer Churn Intelligence")

    st.markdown(
        "### Understand customers. Predict churn. Improve retention."
    )

    st.markdown("---")

    total_customers = len(df)

    churned = (df["Churn"] == "Yes").sum()

    retained = (df["Churn"] == "No").sum()

    churn_rate = (churned / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned:,}"
    )

    col3.metric(
        "Retained Customers",
        f"{retained:,}"
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.markdown("---")

    # CHURN DISTRIBUTION

    left, right = st.columns(2)

    with left:

        st.subheader("Customer Churn Distribution")

        churn_counts = df["Churn"].value_counts()

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            churn_counts,
            labels=churn_counts.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Churn Distribution")

        st.pyplot(fig)

        plt.close(fig)

    with right:

        st.subheader("Churn by Contract Type")

        contract_data = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.barplot(
            data=contract_data,
            x="Contract",
            y="Churn Rate",
            ax=ax
        )

        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Contract Type")
        ax.set_title("Contract-wise Churn Rate")

        plt.xticks(rotation=15)

        st.pyplot(fig)

        plt.close(fig)

    st.markdown("---")

    # CUSTOMER DATA PREVIEW

    st.subheader("Customer Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.success(
        "Customer Churn Intelligence Dashboard loaded successfully!"
    )


# ==========================================
# CUSTOMER ANALYSIS
# ==========================================

elif page == "Customer Analysis":

    st.title("👥 Customer Analysis")

    st.markdown(
        "Explore customer behaviour using interactive filters."
    )

    st.markdown("---")

    # FILTERS

    st.subheader("Filter Customers")

    col1, col2, col3 = st.columns(3)

    with col1:
        contract_filter = st.selectbox(
            "Contract Type",
            ["All"] + sorted(df["Contract"].unique().tolist())
        )

    with col2:
        internet_filter = st.selectbox(
            "Internet Service",
            ["All"] + sorted(df["InternetService"].unique().tolist())
        )

    with col3:
        churn_filter = st.selectbox(
            "Churn Status",
            ["All", "Yes", "No"]
        )

    filtered_df = df.copy()

    if contract_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Contract"] == contract_filter
        ]

    if internet_filter != "All":
        filtered_df = filtered_df[
            filtered_df["InternetService"] == internet_filter
        ]

    if churn_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Churn"] == churn_filter
        ]

    st.markdown("---")

    # CUSTOMER METRICS

    st.subheader("Customer Statistics")

    total = len(filtered_df)

    churned = (filtered_df["Churn"] == "Yes").sum()

    retained = (filtered_df["Churn"] == "No").sum()

    churn_rate = (churned / total * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", f"{total:,}")
    col2.metric("Churned", f"{churned:,}")
    col3.metric("Retained", f"{retained:,}")
    col4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.markdown("---")

    if total > 0:

        # TENURE ANALYSIS

        left, right = st.columns(2)

        with left:

            st.subheader("Customer Tenure Analysis")

            fig, ax = plt.subplots(figsize=(7, 4))

            sns.histplot(
                data=filtered_df,
                x="tenure",
                hue="Churn",
                bins=20,
                kde=True,
                ax=ax
            )

            ax.set_xlabel("Tenure (Months)")
            ax.set_ylabel("Number of Customers")

            st.pyplot(fig)
            plt.close(fig)

        # MONTHLY CHARGES ANALYSIS

        with right:

            st.subheader("Monthly Charges by Churn")

            fig, ax = plt.subplots(figsize=(7, 4))

            sns.boxplot(
                data=filtered_df,
                x="Churn",
                y="MonthlyCharges",
                ax=ax
            )

            ax.set_xlabel("Churn Status")
            ax.set_ylabel("Monthly Charges")

            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # CUSTOMER TABLE

        st.subheader("Filtered Customer Records")

        st.dataframe(
            filtered_df.drop(columns=["customerID"]).head(100),
            use_container_width=True
        )

        st.caption("Showing up to 100 matching customer records.")

    else:

        st.warning("No customers found for the selected filters.")


# ==========================================
# CHURN PREDICTION
# ==========================================

elif page == "Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown(
        "Enter customer details to predict the probability of churn."
    )

    st.markdown("---")

    with st.form("churn_prediction_form"):

        st.subheader("Customer Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender", ["Female", "Male"]
            )

            senior = st.selectbox(
                "Senior Citizen", [0, 1]
            )

            partner = st.selectbox(
                "Partner", ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents", ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (Months)", 0, 72, 12
            )

        with col2:
            phone = st.selectbox(
                "Phone Service", ["Yes", "No"]
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        st.markdown("---")

        st.subheader("Subscription Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

        with col2:
            paperless = st.selectbox(
                "Paperless Billing", ["Yes", "No"]
            )

        with col3:
            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        col1, col2 = st.columns(2)

        with col1:
            monthly = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

        with col2:
            total = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        try:

            model = joblib.load("models/random_forest.pkl")

            preprocessor = joblib.load(
                "models/preprocessor.pkl"
            )

            processed_customer = preprocessor.transform(
                customer
            )

            prediction = model.predict(
                processed_customer
            )[0]

            probability = model.predict_proba(
                processed_customer
            )[0][1]

            churn_probability = probability * 100
            st.session_state["last_customer"] = customer.copy()
            st.session_state["last_probability"] = probability
            st.session_state["last_prediction"] = prediction
            st.markdown("---")

            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2f}%"
                )

            with col2:
                if churn_probability < 30:
                    risk = "LOW"
                elif churn_probability < 60:
                    risk = "MEDIUM"
                else:
                    risk = "HIGH"

                st.metric(
                    "Churn Risk",
                    risk
                )

            if prediction == 1:

                st.error(
                    "⚠️ Model Prediction: Customer is likely to churn."
                )

            else:

                st.success(
                    "✅ Model Prediction: Customer is likely to stay."
                )

            st.progress(float(probability))

            st.caption(
                "Risk levels use project-defined probability thresholds: "
                "below 30% = Low, 30–59.99% = Medium, "
                "60% and above = High."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")

            # ==========================================
# EXPLAINABLE AI - SHAP
# ==========================================

elif page == "Explainable AI":

    st.title("🧠 Explainable AI")

    st.markdown(
        "Understand which customer characteristics influence "
        "the model's churn prediction."
    )

    st.markdown("---")

    if "last_customer" not in st.session_state:

        st.warning(
            "Please visit Churn Prediction and predict a customer first."
        )

    else:

        import shap
        import numpy as np

        customer = st.session_state["last_customer"]

        model = joblib.load("models/random_forest.pkl")

        preprocessor = joblib.load("models/preprocessor.pkl")

        processed = preprocessor.transform(customer)

        explainer = shap.TreeExplainer(model)

        raw_values = explainer.shap_values(processed)

        if isinstance(raw_values, list):

            shap_values = raw_values[1][0]

        else:

            shap_array = np.asarray(raw_values)

            if shap_array.ndim == 3:

                shap_values = shap_array[0, :, 1]

            else:

                shap_values = shap_array[0]

        feature_names = preprocessor.get_feature_names_out()

        explanation_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_values
        })

        explanation_df["Feature"] = (
            explanation_df["Feature"]
            .str.replace("numerical__", "", regex=False)
            .str.replace("categorical__", "", regex=False)
        )

        explanation_df = explanation_df.sort_values(
            "SHAP Value",
            ascending=False
        )

        probability = st.session_state["last_probability"] * 100

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        col2.metric(
            "Predicted Status",
            "Likely to Churn"
            if st.session_state["last_prediction"] == 1
            else "Likely to Stay"
        )

        st.markdown("---")

        st.subheader("Factors Increasing Churn")

        positive = explanation_df[
            explanation_df["SHAP Value"] > 0
        ].head(10)

        if not positive.empty:

            st.bar_chart(
                positive.set_index("Feature")["SHAP Value"]
            )

        else:

            st.info("No positive churn contributions found.")

        st.markdown("---")

        st.subheader("Factors Decreasing Churn")

        negative = explanation_df[
            explanation_df["SHAP Value"] < 0
        ].sort_values("SHAP Value").head(10)

        if not negative.empty:

            st.bar_chart(
                negative.set_index("Feature")["SHAP Value"]
            )

        else:

            st.info("No negative churn contributions found.")

        st.markdown("---")

        st.subheader("Complete Feature Explanation")

        st.dataframe(
            explanation_df,
            use_container_width=True
        )

        st.info(
            "Positive SHAP values push the model output towards churn. "
            "Negative values push it away from churn. "
            "These are model explanations, not proof of causation."
        )


# ==========================================
# RETENTION RECOMMENDATIONS
# ==========================================

elif page == "Retention Recommendations":

    st.title("💡 Retention Recommendations")

    st.markdown(
        "Identify possible customer retention actions "
        "based on customer information."
    )

    st.markdown("---")

    if "last_customer" not in st.session_state:

        st.warning(
            "Please make a prediction in the Churn Prediction page first."
        )

    else:

        customer = st.session_state["last_customer"].iloc[0]

        probability = st.session_state["last_probability"] * 100

        recommendations = []

        if customer["Contract"] == "Month-to-month":

            recommendations.append((
                "Long-term Contract Offer",
                "Offer discounted one-year or two-year plans "
                "to encourage long-term subscriptions."
            ))

        if customer["tenure"] < 12:

            recommendations.append((
                "New Customer Onboarding",
                "Provide onboarding assistance, welcome offers "
                "and regular follow-ups."
            ))

        if customer["OnlineSecurity"] == "No":

            recommendations.append((
                "Online Security Package",
                "Offer online security services with suitable "
                "pricing and explain their benefits."
            ))

        if customer["TechSupport"] == "No":

            recommendations.append((
                "Technical Support",
                "Provide accessible technical support "
                "to improve the customer experience."
            ))

        if customer["InternetService"] == "Fiber optic":

            recommendations.append((
                "Internet Service Review",
                "Review service quality, pricing and customer "
                "satisfaction for fiber optic users."
            ))

        if customer["PaymentMethod"] == "Electronic check":

            recommendations.append((
                "Payment Convenience",
                "Offer convenient automatic payment options "
                "and explain their benefits."
            ))

        if customer["MonthlyCharges"] > 70:

            recommendations.append((
                "Pricing Review",
                "Consider suitable discounts or alternative "
                "plans for customers with higher monthly charges."
            ))

        if probability >= 60:

            recommendations.append((
                "High-Priority Retention",
                "Prioritize a personal follow-up and understand "
                "the customer's concerns."
            ))

        elif probability >= 30:

            recommendations.append((
                "Preventive Retention",
                "Monitor customer satisfaction and provide "
                "relevant retention offers."
            ))

        else:

            recommendations.append((
                "Customer Engagement",
                "Continue providing good service and periodically "
                "monitor customer satisfaction."
            ))

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability:.2f}%"
        )

        if probability < 30:
            risk = "LOW"
        elif probability < 60:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        col2.metric("Risk Category", risk)

        st.markdown("---")

        st.subheader("Recommended Retention Actions")

        for index, recommendation in enumerate(
            recommendations, start=1
        ):

            with st.container(border=True):

                st.markdown(
                    f"### {index}. {recommendation[0]}"
                )

                st.write(recommendation[1])

        st.markdown("---")

        st.caption(
            "Recommendations are rule-based suggestions. "
            "Their effectiveness should be validated using "
            "actual customer retention outcomes."
        )
# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.title("📈 Model Performance")

    st.markdown(
        "### Evaluate, compare and understand machine learning models."
    )

    st.markdown("---")

    # ======================================
    # MODEL METRICS
    # ======================================

    metrics = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Tuned Random Forest"
        ],

        "Accuracy": [
            80.55, 73.95, 76.15, 80.34
        ],

        "Precision": [
            65.72, 50.60, 53.51, 67.51
        ],

        "Recall": [
            55.88, 78.61, 77.54, 50.00
        ],

        "F1 Score": [
            60.40, 61.57, 63.32, 57.45
        ],

        "ROC-AUC": [
            84.21, 82.95, 84.12, 84.34
        ]

    })

    # ======================================
    # PERFORMANCE OVERVIEW
    # ======================================

    st.subheader("🏆 Performance Overview")

    best_accuracy = metrics.loc[
        metrics["Accuracy"].idxmax()
    ]

    best_f1 = metrics.loc[
        metrics["F1 Score"].idxmax()
    ]

    best_recall = metrics.loc[
        metrics["Recall"].idxmax()
    ]

    best_roc = metrics.loc[
        metrics["ROC-AUC"].idxmax()
    ]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Accuracy",
        f'{best_accuracy["Accuracy"]:.2f}%',
        best_accuracy["Model"]
    )

    col2.metric(
        "Best Precision",
        f'{metrics["Precision"].max():.2f}%',
        metrics.loc[
            metrics["Precision"].idxmax(), "Model"
        ]
    )

    col3.metric(
        "Best Recall",
        f'{best_recall["Recall"]:.2f}%',
        best_recall["Model"]
    )

    col4.metric(
        "Best F1 Score",
        f'{best_f1["F1 Score"]:.2f}%',
        best_f1["Model"]
    )

    st.markdown("---")

    # ======================================
    # COMPLETE PERFORMANCE TABLE
    # ======================================

    st.subheader("📋 Holdout Test Performance")

    st.write(
        "Performance of four machine learning models "
        "on the same independent test dataset."
    )

    st.dataframe(
        metrics.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }).highlight_max(
            subset=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],
            color="#c7f9e9"
        ),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================
    # ACCURACY COMPARISON
    # ======================================

    st.subheader("🎯 Accuracy Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metrics,
        x="Accuracy",
        y="Model",
        ax=ax
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("Accuracy (%)")

    ax.set_ylabel("Machine Learning Model")

    ax.set_title(
        "Accuracy of Different Models",
        fontsize=14,
        fontweight="bold"
    )

    ax.bar_label(
        ax.containers[0],
        fmt="%.2f%%",
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # ======================================
    # PRECISION COMPARISON
    # ======================================

    st.subheader("🎯 Precision Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metrics,
        x="Precision",
        y="Model",
        ax=ax
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("Precision (%)")

    ax.set_ylabel("Machine Learning Model")

    ax.set_title(
        "Precision of Different Models",
        fontsize=14,
        fontweight="bold"
    )

    ax.bar_label(
        ax.containers[0],
        fmt="%.2f%%",
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # ======================================
    # RECALL COMPARISON
    # ======================================

    st.subheader("🎯 Recall Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metrics,
        x="Recall",
        y="Model",
        ax=ax
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("Recall (%)")

    ax.set_ylabel("Machine Learning Model")

    ax.set_title(
        "Recall of Different Models",
        fontsize=14,
        fontweight="bold"
    )

    ax.bar_label(
        ax.containers[0],
        fmt="%.2f%%",
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # ======================================
    # F1 SCORE COMPARISON
    # ======================================

    st.subheader("📊 F1 Score Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metrics,
        x="F1 Score",
        y="Model",
        ax=ax
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("F1 Score (%)")

    ax.set_ylabel("Machine Learning Model")

    ax.set_title(
        "F1 Score of Different Models",
        fontsize=14,
        fontweight="bold"
    )

    ax.bar_label(
        ax.containers[0],
        fmt="%.2f%%",
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # ======================================
    # ROC-AUC COMPARISON
    # ======================================

    st.subheader("📈 ROC-AUC Comparison")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metrics,
        x="ROC-AUC",
        y="Model",
        ax=ax
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("ROC-AUC (%)")

    ax.set_ylabel("Machine Learning Model")

    ax.set_title(
        "ROC-AUC of Different Models",
        fontsize=14,
        fontweight="bold"
    )

    ax.bar_label(
        ax.containers[0],
        fmt="%.2f%%",
        padding=5
    )

    ax.grid(axis="x", alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.markdown("---")

    # ======================================
    # CROSS VALIDATION
    # ======================================

    st.subheader("🔄 Cross-Validation Results")

    st.write(
        "Five-fold stratified cross-validation results "
        "using the training dataset."
    )

    cv_metrics = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "Accuracy": [
            80.21, 73.54, 76.11
        ],

        "Precision": [
            65.29, 50.20, 53.52
        ],

        "Recall": [
            54.31, 79.60, 76.12
        ],

        "F1 Score": [
            59.23, 61.53, 62.84
        ],

        "ROC-AUC": [
            84.62, 82.55, 84.52
        ],

        "F1 Standard Deviation": [
            2.99, 1.97, 1.83
        ]

    })

    st.dataframe(
        cv_metrics.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%",
            "F1 Standard Deviation": "{:.2f}%"
        }),
        hide_index=True,
        use_container_width=True
    )

    st.markdown("---")

    # ======================================
    # MODEL EVALUATION SUMMARY
    # ======================================

    st.subheader("📝 Model Evaluation Summary")

    st.success(
        "Random Forest is selected as the primary "
        "prediction model based on its holdout and "
        "cross-validation F1 performance."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Best Performing Models")

        st.write("🏆 Accuracy: Logistic Regression — 80.55%")

        st.write("🎯 Precision: Tuned Random Forest — 67.51%")

        st.write("📈 Recall: Decision Tree — 78.61%")

        st.write("⭐ Holdout F1: Random Forest — 63.32%")

        st.write("🔄 CV F1: Random Forest — 62.84%")

    with col2:

        st.markdown("### Final Model Selection")

        st.write(
            "**Selected Model:** Random Forest"
        )

        st.write(
            "**Holdout F1 Score:** 63.32%"
        )

        st.write(
            "**Cross-Validation F1 Score:** 62.84%"
        )

        st.write(
            "**ROC-AUC:** 84.12%"
        )

    st.markdown("---")

    st.caption(
        "All reported holdout metrics are from the same "
        "independent test split. Cross-validation metrics "
        "are calculated separately using the training set."
    )


# ==========================================
# OTHER MODULES
# ==========================================

else:

    st.title(page)

    st.info("Module under development.")

