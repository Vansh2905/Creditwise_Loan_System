import os
import joblib
import streamlit as st
import pandas as pd

# Load model and feature names using path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
saved = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
model = saved["model"]
FEATURE_NAMES = saved["feature_names"]

st.title("🏦 CreditWise Loan Approval Predictor")
st.markdown("Fill in the applicant details below to predict loan approval.")

col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income", min_value=0, value=10000)
    coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=0)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    dependents = st.selectbox("Dependents", [0, 1, 2, 3])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3, 4])
    dti_ratio = st.slider("DTI Ratio", min_value=0.1, max_value=0.6, value=0.35, step=0.01)
    savings = st.number_input("Savings", min_value=0, value=5000)
    collateral_value = st.number_input("Collateral Value", min_value=0, value=20000)

with col2:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=15000)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84])
    # Label encoded
    education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    # OHE columns (drop='first' means first category is the reference/baseline = 0 for all dummies)
    employment_status = st.selectbox("Employment Status", ["Contract", "Salaried", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    loan_purpose = st.selectbox("Loan Purpose", ["Business", "Car", "Education", "Home", "Personal"])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    employer_category = st.selectbox("Employer Category", ["Business", "Government", "MNC", "Private", "Unemployed"])

if st.button("🔍 Predict Loan Approval", use_container_width=True):
    # Label encoding
    education_level_enc = 0 if education_level == "Graduate" else 1

    # OHE with drop='first':
    emp_salaried = 1 if employment_status == "Salaried" else 0
    emp_self_employed = 1 if employment_status == "Self-employed" else 0
    emp_unemployed = 1 if employment_status == "Unemployed" else 0
    marital_single = 1 if marital_status == "Single" else 0
    loan_car = 1 if loan_purpose == "Car" else 0
    loan_education = 1 if loan_purpose == "Education" else 0
    loan_home = 1 if loan_purpose == "Home" else 0
    loan_personal = 1 if loan_purpose == "Personal" else 0
    prop_semiurban = 1 if property_area == "Semiurban" else 0
    prop_urban = 1 if property_area == "Urban" else 0
    gender_male = 1 if gender == "Male" else 0
    emp_cat_government = 1 if employer_category == "Government" else 0
    emp_cat_mnc = 1 if employer_category == "MNC" else 0
    emp_cat_private = 1 if employer_category == "Private" else 0
    emp_cat_unemployed = 1 if employer_category == "Unemployed" else 0

    input_data = {
        "Applicant_Income": float(applicant_income),
        "Coapplicant_Income": float(coapplicant_income),
        "Age": float(age),
        "Dependents": float(dependents),
        "Existing_Loans": float(existing_loans),
        "Savings": float(savings),
        "Collateral_Value": float(collateral_value),
        "Loan_Amount": float(loan_amount),
        "Loan_Term": float(loan_term),
        "Credit_Score_sq": float(credit_score) ** 2,
        "DTI_Ratio_sq": float(dti_ratio) ** 2,
        "Education_Level": education_level_enc,
        "Employment_Status_Salaried": emp_salaried,
        "Employment_Status_Self-employed": emp_self_employed,
        "Employment_Status_Unemployed": emp_unemployed,
        "Marital_Status_Single": marital_single,
        "Loan_Purpose_Car": loan_car,
        "Loan_Purpose_Education": loan_education,
        "Loan_Purpose_Home": loan_home,
        "Loan_Purpose_Personal": loan_personal,
        "Property_Area_Semiurban": prop_semiurban,
        "Property_Area_Urban": prop_urban,
        "Gender_Male": gender_male,
        "Employer_Category_Government": emp_cat_government,
        "Employer_Category_MNC": emp_cat_mnc,
        "Employer_Category_Private": emp_cat_private,
        "Employer_Category_Unemployed": emp_cat_unemployed,
    }

    input_df = pd.DataFrame([input_data])[FEATURE_NAMES]
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if prediction == 1 or prediction == "Yes":
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #1a7a4a, #25a265);
                border-radius: 16px;
                padding: 36px 24px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(37,162,101,0.4);
            ">
                <div style="font-size: 64px; margin-bottom: 8px;">✅</div>
                <div style="color: white; font-size: 28px; font-weight: 700; margin-bottom: 8px;">Loan Approved!</div>
                <div style="color: rgba(255,255,255,0.85); font-size: 15px;">Congratulations! Based on the provided details, your loan application is likely to be approved.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #8b1a1a, #c0392b);
                border-radius: 16px;
                padding: 36px 24px;
                text-align: center;
                box-shadow: 0 4px 20px rgba(192,57,43,0.4);
            ">
                <div style="font-size: 64px; margin-bottom: 8px;">❌</div>
                <div style="color: white; font-size: 28px; font-weight: 700; margin-bottom: 8px;">Loan Not Approved</div>
                <div style="color: rgba(255,255,255,0.85); font-size: 15px;">Based on the provided details, your loan application is unlikely to be approved at this time.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
