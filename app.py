import streamlit as st
import pandas as pd
import joblib

model = joblib.load("loan_model.pkl")
features = joblib.load("features.pkl")

st.title("Loan Approval Prediction")
st.write("Enter the applicant's details to predict whether the loan will be approved.")

col1, col2 = st.columns(2)
with col1:
    dependents = st.number_input("Number of dependents", 0, 5, 0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_emp = st.selectbox("Self employed", ["No", "Yes"])
    income = st.number_input("Annual income", 200000, 9900000, 5000000, step=100000)
    loan_amount = st.number_input("Loan amount", 300000, 39500000, 15000000, step=100000)
    loan_term = st.number_input("Loan term (years)", 2, 20, 10)
with col2:
    cibil = st.slider("CIBIL score", 300, 900, 650)
    res = st.number_input("Residential assets value", 0, 29100000, 5000000, step=100000)
    com = st.number_input("Commercial assets value", 0, 19400000, 5000000, step=100000)
    lux = st.number_input("Luxury assets value", 300000, 39200000, 15000000, step=100000)
    bank = st.number_input("Bank asset value", 0, 14700000, 5000000, step=100000)

if st.button("Predict"):
    total_assets = res + com + lux + bank
    annual_repayment = loan_amount / loan_term
    row = {
        "no_of_dependents": dependents,
        "education": 1 if education == "Graduate" else 0,
        "self_employed": 1 if self_emp == "Yes" else 0,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "residential_assets_value": res,
        "commercial_assets_value": com,
        "luxury_assets_value": lux,
        "bank_asset_value": bank,
        "total_assets": total_assets,
        "loan_to_income": loan_amount / income,
        "assets_to_loan": total_assets / loan_amount,
        "annual_repayment": annual_repayment,
        "repayment_to_income": annual_repayment / income,
    }
    X_new = pd.DataFrame([row])[features]
    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.success(f"Loan likely APPROVED (probability {prob:.1%})")
    else:
        st.error(f"Loan likely REJECTED (approval probability {prob:.1%})")