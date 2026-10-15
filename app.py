
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "churn_pipeline.joblib"

st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer may leave.")

model = joblib.load(MODEL_PATH)

st.subheader("Enter Customer Details")

tenure = st.number_input("Tenure (months)", 0, 100, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 100000.0, 840.0)

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)
internet = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
backup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
protection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
tv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
movies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)
contract = st.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

if st.button("Predict Churn"):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(customer)[0][1]
        st.metric("Churn Probability", f"{probability:.1%}")

    if str(prediction).lower() in ["yes", "1", "true"]:
        st.warning("Customer is predicted to churn.")
    else:
        st.success("Customer is predicted to stay.")
