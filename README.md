# 🏦 AI-Based Smart Credit Risk Analysis System

An intelligent credit approval and risk assessment platform developed using Machine Learning and Streamlit.

This project predicts whether a customer's loan application should be approved by analyzing financial and demographic information. Multiple machine learning algorithms are integrated and compared within an interactive dashboard.

---

## 🚀 Project Overview

Financial institutions process thousands of loan applications every day. Evaluating these applications manually can be slow, inconsistent, and prone to human error.

This project aims to support credit decision-making by utilizing machine learning models trained on historical loan application data.

The system provides:

* Credit approval prediction
* Risk score estimation
* Multi-model comparison
* Explainable AI visualizations
* Interactive Streamlit dashboard

---

## 📊 Dataset

The project uses a loan approval dataset containing applicant demographic and financial information.

### Features

| Feature           | Description                 |
| ----------------- | --------------------------- |
| Gender            | Applicant Gender            |
| Married           | Marital Status              |
| Dependents        | Number of Dependents        |
| Education         | Education Level             |
| Self_Employed     | Employment Status           |
| ApplicantIncome   | Monthly Income              |
| CoapplicantIncome | Co-Applicant Income         |
| LoanAmount        | Requested Loan Amount       |
| Loan_Amount_Term  | Loan Duration               |
| Credit_History    | Previous Credit Performance |
| Property_Area     | Urban / Rural Information   |

### Target Variable

| Variable    | Meaning                     |
| ----------- | --------------------------- |
| Loan_Status | Approved (Y) / Rejected (N) |

---

## 🤖 Machine Learning Models

The following algorithms were trained and evaluated:

### 1. Logistic Regression

* Accuracy: 78.8%
* Recall: 98.7%
* F1 Score: 85.8%
* AUC: 0.739

### 2. Decision Tree

* Accuracy: 69.9%
* Recall: 77.5%
* F1 Score: 77.0%

### 3. Random Forest

* Accuracy: 76.4%
* Recall: 95.0%
* F1 Score: 83.9%

### 4. Ensemble Model

Average probability generated from:

* Logistic Regression
* Decision Tree
* Random Forest

Used as the final recommendation model.

---

## 🖥️ Streamlit Dashboard Features

### Single Customer Analysis

Users can enter:

* Gender
* Marital Status
* Education
* Applicant Income
* Co-Applicant Income
* Loan Amount
* Loan Term
* Credit History

The system instantly predicts:

* Approval Probability
* Risk Level
* Model Recommendations

---

### Multi-Model Prediction

The dashboard displays predictions from:

* Logistic Regression
* Decision Tree
* Random Forest
* Ensemble Model

allowing direct comparison between models.

---

### Explainable AI (XAI)

Feature contribution analysis highlights the most influential variables in the credit decision process.

Important factors include:

* Credit History
* Applicant Income
* Loan Amount

---

### Model Performance Visualization

The dashboard includes:

* Accuracy Comparison
* ROC Curve
* Feature Importance Charts
* Performance Metrics

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-Learn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Web Interface

* Streamlit

### Model Persistence

* Joblib

---

## 📁 Project Structure

```text
AI-Credit-Risk-Analysis/
│
├── data/
│   └── loan_data.csv
│
├── models/
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── app.py
├── loan_prediction_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/ErbolZhanyshov/credit-risk-analysis-system.git
cd AI-Credit-Risk-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

---

## 🎓 Academic Information

Developed as a Machine Learning and Artificial Intelligence course project.

**Department:** Computer Engineering

**University:** Hitit University

---

## 📌 Future Improvements

* SHAP Integration
* Bulk Customer Prediction
* Excel Report Export
* Real-Time Model Monitoring
* Advanced Risk Scoring
* Deep Learning Models
* Cloud Deployment

---

## 👨‍💻 Authors

Developed by:

**Erbol Zhanyshov**

Hitit University – Computer Engineering
