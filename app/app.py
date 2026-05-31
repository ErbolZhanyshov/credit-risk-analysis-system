import streamlit as st

st.set_page_config(
    page_title="Credit Risk Analysis System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Risk Analysis System")

st.markdown("""
Bu sistem müşterinin kredi başvurusunu analiz ederek
kredi onayı alma olasılığını tahmin eder.
""")

st.divider()

st.subheader("Customer Information")

#----------------------------------------------------------------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Married",
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
    ["Yes", "No"]
)

#-----------------------------------------------------------------

applicant_income = st.number_input(
    "Applicant Income",
    min_value=0
)

coapplicant_income = st.number_input(
    "Coapplicant Income",
    min_value=0
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_amount_term = st.number_input(
    "Loan Amount Term",
    min_value=0
)

credit_history = st.selectbox(
    "Credit History",
    [1, 0]
)

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)

#----------------------------------------------------------------------------

if st.button("Predict Loan Approval"):

    st.success(
        "Model integration will be added in the next step."
    )