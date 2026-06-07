import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Credit Risk Analysis System",
    page_icon="🏦",
    layout="wide"
)

# =====================================================
# LOAD MODELS
# =====================================================

lr_model = joblib.load("models/logistic_regression_model.pkl")
dt_model = joblib.load("models/decision_tree_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #f4f6f9;
}

.main-title{
    font-size:38px;
    font-weight:700;
    color:#1a2a6c;
}

.sub-title{
    color:#6c757d;
    font-size:16px;
}

.metric-card{
    background:white;
    border-radius:16px;
    padding:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
    text-align:center;
    border-top:5px solid #1a2a6c;
}

.metric-value{
    font-size:32px;
    font-weight:bold;
    color:#1a2a6c;
}

.metric-label{
    color:#6c757d;
}

.sidebar-card{
    background:white;
    padding:15px;
    border-radius:12px;
            
.stSelectbox,
.stNumberInput,
.stSlider{
    background:white;
    border-radius:10px;}
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🏦 Navigation")

    menu = st.radio(
        "Menu",
        [
            "Dashboard",
            "Single Prediction",
            "Bulk Prediction",
            "Model Analytics",
            "About"
        ]
    )

    st.markdown("---")

    st.subheader("ℹ️ About")

    st.info("""
    AI Credit Risk Analysis System

    Hitit University

    Computer Engineering
    """)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="main-title">
🏦 AI Credit Risk Analysis System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sub-title">
Multi-Model Banking Decision Support Platform
</div>
""", unsafe_allow_html=True)

st.write("")

# =====================================================
# SINGLE CUSTOMER ANALYSIS
# =====================================================

st.markdown("## 🧑‍💼 Single Customer Credit Analysis")

st.caption(
    "Enter customer financial information below to evaluate credit approval probability using multiple machine learning models."
)

st.write("")

# =====================================================
# FORM CARDS
# =====================================================

col1, col2, col3 = st.columns(3)

# -------------------------------------------------
# CUSTOMER PROFILE
# -------------------------------------------------

with col1:

    st.markdown("### 👤 Customer Profile")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0", "1", "2", "3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["No", "Yes"]
    )

# -------------------------------------------------
# FINANCIAL PROFILE
# -------------------------------------------------

with col2:

    st.markdown("### 💰 Financial Profile")

    applicant_income = st.number_input(
        "Applicant Income",
        value=5400,
        step=500
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        value=0,
        step=500
    )

    loan_amount = st.slider(
        "Loan Amount",
        10,
        500,
        140
    )

    loan_term = st.selectbox(
        "Loan Term (Months)",
        [360, 240, 180, 120, 84]
    )

# -------------------------------------------------
# CREDIT INFORMATION
# -------------------------------------------------

with col3:

    st.markdown("### 🏦 Credit Information")

    credit_history = st.selectbox(
        "Credit History",
        [
            "Good (1.0)",
            "Bad (0.0)"
        ]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Urban",
            "Semiurban",
            "Rural"
        ]
    )

st.write("")

analyze = st.button(
    "🚀 Analyze Credit Risk"
)

# =====================================================
# MODEL ANALYSIS
# =====================================================

if analyze:

    with st.spinner("Analyzing customer credit risk..."):
        time.sleep(1)

    # ----------------------------------
    # ENCODING
    # ----------------------------------

    gender_v = 1 if gender == "Male" else 0
    married_v = 1 if married == "Yes" else 0

    if dependents == "3+":
        dependents_v = 3
    else:
        dependents_v = int(dependents)

    education_v = 0 if education == "Graduate" else 1
    self_emp_v = 1 if self_employed == "Yes" else 0

    credit_v = 1 if credit_history == "Good (1.0)" else 0

    property_map = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }

    property_v = property_map[property_area]

    # ----------------------------------
    # FEATURE VECTOR
    # ----------------------------------

    features = np.array([[
        gender_v,
        married_v,
        dependents_v,
        education_v,
        self_emp_v,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_v,
        property_v
    ]])

    features_scaled = scaler.transform(features)

    # Logistic Regression
    lr_prob = lr_model.predict_proba(features_scaled)[0][1]
    lr_pred = lr_model.predict(features_scaled)[0]

    # Decision Tree - ARTIK DÜZGÜN ÇALIŞACAK
    dt_prob = dt_model.predict_proba(features_scaled)[0][1]
    dt_pred = dt_model.predict(features_scaled)[0]

    # Random Forest - ARTIK DÜZGÜN ÇALIŞACAK
    rf_prob = rf_model.predict_proba(features_scaled)[0][1]
    rf_pred = rf_model.predict(features_scaled)[0]

    ensemble_prob = (
        lr_prob +
        dt_prob +
        rf_prob
    ) / 3

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")

    # =====================================================
    # GAUGE CARDS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    models = {
        "Logistic Regression": lr_prob,
        "Decision Tree": dt_prob,
        "Random Forest": rf_prob,
        "Ensemble": ensemble_prob
    }

    columns = [col1, col2, col3, col4]

    for (name, prob), col in zip(models.items(), columns):

        with col:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={'suffix': "%"},
                    title={'text': name},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'thickness': 0.3},
                        'steps': [
                            {'range': [0, 50], 'color': "#ff4d4f"},
                            {'range': [50, 75], 'color': "#fadb14"},
                            {'range': [75, 100], 'color': "#52c41a"}
                        ]
                    }
                )
            )

            fig.update_layout(
                height=250,
                margin=dict(l=10,r=10,t=50,b=10)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if prob >= 0.5:

                st.success(
                    f"Low Risk\n\nApproval Probability: {prob*100:.1f}%"
                )

            else:

                st.error(
                    f"High Risk\n\nRisk Score: {(1-prob)*100:.1f}%"
                )

    # =====================================================
    # FINAL DECISION
    # =====================================================

    st.markdown("---")
    st.markdown("## 🏦 Final AI Decision")

    if ensemble_prob >= 0.50:

        st.success(
            f"""
            CREDIT APPROVED

            Ensemble Confidence Score:
            {ensemble_prob*100:.2f}%
            """
        )

        st.info(
            """
            AI Recommendation:

            • Credit history is positive

            • Income level supports repayment

            • Customer is considered low risk

            • Loan approval is recommended
            """
        )

    else:

        st.error(
            f"""
            CREDIT REJECTED

            Risk Score:
            {(1-ensemble_prob)*100:.2f}%
            """
        )

        st.warning(
            """
            AI Recommendation:

            • Credit history is insufficient

            • Risk level is high

            • Additional collateral may be required

            • Manual review is recommended
            """
        )

    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================

    st.markdown("---")
    st.markdown("## 👤 Customer Summary")

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric(
            "Applicant Income",
            f"${applicant_income:,}"
        )

    with s2:
        st.metric(
            "Coapplicant Income",
            f"${coapplicant_income:,}"
        )

    with s3:
        st.metric(
            "Loan Amount",
            f"${loan_amount:,}"
        )

    with s4:
        st.metric(
            "Loan Term",
            f"{loan_term} Months"
        )
