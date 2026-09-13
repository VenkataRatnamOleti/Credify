import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credify",
    page_icon="💳",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    bundle = joblib.load("creditwise_model.pkl")

    return bundle


bundle = load_model()

model = bundle["model"]
scaler = bundle["scaler"]
ohe = bundle["ohe"]
education_encoder = bundle["education_encoder"]
target_encoder = bundle["target_encoder"]
feature_columns = bundle["feature_columns"]


# =========================================================
# HEADER
# =========================================================

st.title("💳 Credify")

st.subheader("AI-Powered Loan Approval Prediction")

st.write(
    "Enter the applicant's financial and personal details "
    "to predict whether the loan is likely to be approved."
)

st.divider()


# =========================================================
# APPLICANT INFORMATION
# =========================================================

st.header("👤 Applicant Information")

col1, col2, col3 = st.columns(3)


with col1:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        max_value=100000.0,
        value=10000.0,
        step=500.0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35,
        step=1
    )

    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=700.0,
        step=1.0
    )


with col2:

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        max_value=100000.0,
        value=5000.0,
        step=500.0
    )

    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0,
        max_value=20,
        value=2,
        step=1
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    savings = st.number_input(
        "Savings",
        min_value=0.0,
        max_value=200000.0,
        value=10000.0,
        step=500.0
    )


with col3:

    collateral_value = st.number_input(
        "Collateral Value",
        min_value=0.0,
        max_value=200000.0,
        value=25000.0,
        step=500.0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        max_value=200000.0,
        value=20000.0,
        step=500.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1,
        max_value=120,
        value=48,
        step=12
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "Graduate",
            "Not Graduate"
        ]
    )


# =========================================================
# CATEGORICAL INFORMATION
# =========================================================

st.divider()

st.header("📋 Applicant Details")

col1, col2, col3 = st.columns(3)


with col1:

    employment_status = st.selectbox(
        "Employment Status",
        [
            "Salaried",
            "Self-employed",
            "Unemployed"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Married",
            "Single"
        ]
    )


with col2:

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Car",
            "Education",
            "Home",
            "Personal",
            "Business"
        ]
    )

    property_area = st.selectbox(
        "Property Area",
        [
            "Rural",
            "Semiurban",
            "Urban"
        ]
    )


with col3:

    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )

    employer_category = st.selectbox(
        "Employer Category",
        [
            "Government",
            "MNC",
            "Private",
            "Unemployed"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.divider()

predict_button = st.button(
    "🔍 Predict Loan Approval",
    type="primary",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_df = pd.DataFrame({

            "Applicant_Income": [applicant_income],

            "Coapplicant_Income": [coapplicant_income],

            "Employment_Status": [employment_status],

            "Age": [age],

            "Marital_Status": [marital_status],

            "Dependents": [dependents],

            "Credit_Score": [credit_score],

            "Existing_Loans": [existing_loans],

            "DTI_Ratio": [dti_ratio],

            "Savings": [savings],

            "Collateral_Value": [collateral_value],

            "Loan_Amount": [loan_amount],

            "Loan_Term": [loan_term],

            "Loan_Purpose": [loan_purpose],

            "Property_Area": [property_area],

            "Education_Level": [education_level],

            "Gender": [gender],

            "Employer_Category": [employer_category]
        })


        # -------------------------------------------------
        # EDUCATION LABEL ENCODING
        # -------------------------------------------------

        input_df["Education_Level"] = (
            education_encoder.transform(
                input_df["Education_Level"]
            )
        )


        # -------------------------------------------------
        # ONE HOT ENCODING
        # -------------------------------------------------

        encoded_input = ohe.transform(
            input_df[
                [
                    "Employment_Status",
                    "Marital_Status",
                    "Loan_Purpose",
                    "Property_Area",
                    "Gender",
                    "Employer_Category"
                ]
            ]
        )


        encoded_input_df = pd.DataFrame(
            encoded_input,
            columns=ohe.get_feature_names_out(
                [
                    "Employment_Status",
                    "Marital_Status",
                    "Loan_Purpose",
                    "Property_Area",
                    "Gender",
                    "Employer_Category"
                ]
            )
        )


        # -------------------------------------------------
        # COMBINE DATA
        # -------------------------------------------------

        input_df = pd.concat(
            [
                input_df.drop(
                    columns=[
                        "Employment_Status",
                        "Marital_Status",
                        "Loan_Purpose",
                        "Property_Area",
                        "Gender",
                        "Employer_Category"
                    ]
                ).reset_index(drop=True),

                encoded_input_df
            ],
            axis=1
        )


        # -------------------------------------------------
        # FEATURE ENGINEERING
        # -------------------------------------------------

        input_df["DTI_Ratio_sq"] = (
            input_df["DTI_Ratio"] ** 2
        )

        input_df["Credit_Score_sq"] = (
            input_df["Credit_Score"] ** 2
        )


        # -------------------------------------------------
        # REMOVE ORIGINAL FEATURES
        # -------------------------------------------------

        input_df = input_df.drop(
            columns=[
                "Credit_Score",
                "DTI_Ratio"
            ]
        )


        # -------------------------------------------------
        # ENSURE EXACT FEATURE ORDER
        # -------------------------------------------------

        input_df = input_df[
            feature_columns
        ]


        # -------------------------------------------------
        # SCALE INPUT
        # -------------------------------------------------

        input_scaled = scaler.transform(
            input_df
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_scaled
        )[0]


        probability = model.predict_proba(
            input_scaled
        )[0]


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.divider()

        st.header("📊 Prediction Result")


        if prediction == 1:

            st.success(
                "### ✅ Loan Likely to be Approved"
            )

            st.write(
                "Based on the provided information, "
                "the model predicts that the loan is likely "
                "to be approved."
            )

        else:

            st.error(
                "### ❌ Loan Likely to be Rejected"
            )

            st.write(
                "Based on the provided information, "
                "the model predicts that the loan is likely "
                "to be rejected."
            )


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Approval Probability",
                f"{probability[1] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Rejection Probability",
                f"{probability[0] * 100:.2f}%"
            )


        # -------------------------------------------------
        # PROGRESS BAR
        # -------------------------------------------------

        st.write("Approval Confidence")

        st.progress(
            float(probability[1])
        )


        st.caption(
            "This prediction is generated by a machine "
            "learning model and should not be treated as "
            "a real financial approval decision."
        )


    except Exception as e:

        st.error(
            "Something went wrong while making the prediction."
        )

        st.exception(e)