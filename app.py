import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="centered")

st.title(" Telco Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn or stay.")

@st.cache_resource
def load_artifacts():
    return joblib.load('best_model.pkl')

try:
    artifacts = load_artifacts()
    model = artifacts['model']
    scaler = artifacts['scaler']
    encoders = artifacts['encoders']
    feature_names = artifacts['feature_names']
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

st.header("Customer Details")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
    online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
    tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=tenure * monthly_charges)

if st.button("Predict Churn Risk", type="primary"):
    raw_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([raw_data])

    for col in input_df.columns:
        if col in encoders and col != 'Churn':
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"This customer is at risk of leaving (Churn Risk: {prob*100:.1f}%)")
    else:
        st.success(f"This customer is likely to stay (Retention Probability: {(1-prob)*100:.1f}%)")