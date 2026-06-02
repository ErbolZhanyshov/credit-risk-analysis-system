import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import time

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
# CSS DESIGN (FINTECH DASHBOARD)
# =====================================================
st.markdown("""
<style>

.main { background-color:#f4f6f9; }

.block-container { padding-top:2rem; }

.metric-card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    text-align:center;
}

.result-approved{
    background:linear-gradient(135deg,#eafaf1,#d1f2eb);
    border-left:8px solid #2ecc71;
    padding:25px;
    border-radius:15px;
}

.result-rejected{
    background:linear-gradient(135deg,#fdecea,#fadbd8);
    border-left:8px solid #e74c3c;
    padding:25px;
    border-radius:15px;
}

.stButton > button{
    background:#1a2a6c;
    color:white;
    width:100%;
    height:50px;
    border:none;
    border-radius:10px;
    font-weight:bold;
}

.stButton > button:hover{
    transform:translateY(-2px);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.title("🏦 AI Credit Risk Analysis System")
st.caption("Multi Model Banking AI Dashboard")

# =====================================================
# INPUT FORM
# =====================================================
st.markdown("## 👤 Customer Input")

c1, c2, c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])

with c2:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    loan_term = st.selectbox("Loan Term", [360, 240, 180, 120, 84])

with c3:
    credit_history = st.selectbox(
        "Credit History",
        ["Good (1)", "Bad (0)"]
    )

applicant_income = st.number_input("Applicant Income", 5400, step=500)
coapplicant_income = st.number_input("Coapplicant Income", 0, step=500)
loan_amount = st.slider("Loan Amount", 10, 500, 140)

analyze = st.button("🚀 Analyze Credit Risk")

# =====================================================
# ANALYSIS
# =====================================================
if not analyze:
    st.info("Fill the form and click Analyze.")

if analyze:

    with st.spinner("AI models are processing..."):
        time.sleep(0.8)

    # =========================
    # ENCODING
    # =========================
    gender_v = 1 if gender == "Male" else 0
    married_v = 1 if married == "Yes" else 0
    education_v = 0 if education == "Graduate" else 1
    credit_v = 1 if credit_history == "Good (1)" else 0

    dependents_v = 0
    self_emp_v = 0
    property_v = 1

    # =========================
    # FEATURE VECTOR
    # =========================
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

    # =========================
    # MODEL PREDICTIONS
    # =========================
    lr_pred = lr_model.predict(features_scaled)[0]
    dt_pred = dt_model.predict(features_scaled)[0]
    rf_pred = rf_model.predict(features_scaled)[0]

    lr_prob = lr_model.predict_proba(features_scaled)[0][1]
    dt_prob = dt_model.predict_proba(features_scaled)[0][1]
    rf_prob = rf_model.predict_proba(features_scaled)[0][1]

    ensemble_prob = (lr_prob + dt_prob + rf_prob) / 3
    approved = ensemble_prob >= 0.5

    # =====================================================
    # MODEL RESULTS
    # =====================================================
    st.markdown("## 🤖 Model Outputs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Logistic Regression", f"{lr_prob*100:.1f}%")

    with c2:
        st.metric("Decision Tree", f"{dt_prob*100:.1f}%")

    with c3:
        st.metric("Random Forest", f"{rf_prob*100:.1f}%")

    with c4:
        st.metric("Ensemble", f"{ensemble_prob*100:.1f}%")

    # =====================================================
    # FINAL DECISION
    # =====================================================
    st.markdown("## 🏦 Final Decision")

    if approved:
        st.markdown(f"""
        <div class="result-approved">
        <h2>✅ CREDIT APPROVED</h2>
        <h3>Confidence: {ensemble_prob*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
        <h2>❌ CREDIT REJECTED</h2>
        <h3>Risk: {(1-ensemble_prob)*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # SUMMARY
    # =====================================================
    st.markdown("## 📋 Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Income", applicant_income)
    col2.metric("Loan", loan_amount)
    col3.metric("Term", loan_term)

    # =====================================================
    # GRAPH
    # =====================================================
    st.markdown("## 📊 Model Comparison")

    df = pd.DataFrame({
        "Model": ["LR", "DT", "RF"],
        "Score": [lr_prob, dt_prob, rf_prob]
    })

    fig, ax = plt.subplots()
    ax.bar(df["Model"], df["Score"])
    ax.set_ylabel("Probability")

    st.pyplot(fig)