import streamlit as st
import pickle
import pandas as pd

# Load model and feature names
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["features_names"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# Function to preprocess input dictionary
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    for col, le in encoders.items():
        if col in df.columns:
            if df.loc[0, col] in le.classes_:
                df[col] = le.transform(df[col])
            else:
                st.warning(f"⚠️ Warning: Unknown value '{df.loc[0, col]}' for {col}. Using default encoding 0.")
                df[col] = 0
    df = df[feature_names]
    return df.values

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

st.title("📉 Customer Churn Prediction")
st.caption("💡 Know your customers before they go 🚪")

st.markdown("---")
st.write("🔎 Please input customer details below:")

# Input widgets with emojis
gender = st.selectbox("👤 Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("🎓 Senior Citizen", ["0", "1"])
Partner = st.selectbox("❤️ Partner", ["No", "Yes"])
Dependents = st.selectbox("👶 Dependents", ["No", "Yes"])
tenure = st.number_input("📅 Tenure (in months)", min_value=0, max_value=72, value=1)
PhoneService = st.selectbox("📞 Phone Service", ["No", "Yes"])
MultipleLines = st.selectbox("📱 Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("🔒 Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("💾 Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("🛡️ Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("🧑‍💻 Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("📺 Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("🎬 Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("📃 Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("🧾 Paperless Billing", ["No", "Yes"])
PaymentMethod = st.selectbox("💳 Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("💰 Monthly Charges", min_value=0.0, max_value=1000.0, value=50.0)
TotalCharges = st.number_input("💵 Total Charges", min_value=0.0, max_value=10000.0, value=100.0)

# Collect input data
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
}

# Prediction button
if st.button("🔍 Predict Churn"):
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)
    result = "⚠️ Customer will churn." if prediction[0] == 1 else "✅ Customer will NOT churn."
    st.success(result)
