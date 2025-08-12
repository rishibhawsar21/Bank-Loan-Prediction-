import streamlit as st
import pandas as pd
import pickle

# Load trained Random Forest model
with open("RFC_Model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💰 Bank Loan Prediction App")

st.markdown("""
This app predicts *whether a customer will accept a personal loan*  
based on their profile and financial information.
""")

# Sidebar for inputs
st.sidebar.header("Input Customer Details")

# Input fields based on your dataset features (after dropping ID, ZIP.Code, Personal.Loan)
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    experience = st.sidebar.number_input("Experience (years)", min_value=0, max_value=80, value=5)
    income = st.sidebar.number_input("Annual Income (in $1000)", min_value=0, value=50)
    family = st.sidebar.selectbox("Family Members", [1, 2, 3, 4])
    ccavg = st.sidebar.number_input("Avg. Credit Card Spending (in $1000)", min_value=0.0, value=2.0)
    education = st.sidebar.selectbox("Education Level", [1, 2, 3])  # Assuming 1=Undergrad, 2=Graduate, 3=Advanced/Professional
    mortgage = st.sidebar.number_input("Mortgage Value", min_value=0, value=0)
    securities_account = st.sidebar.selectbox("Securities Account", [0, 1])
    cd_account = st.sidebar.selectbox("CD Account", [0, 1])
    online = st.sidebar.selectbox("Online Banking", [0, 1])
    creditcard = st.sidebar.selectbox("Credit Card User", [0, 1])

    # Create DataFrame
    data = {
        "Age": age,
        "Experience": experience,
        "Income": income,
        "Family": family,
        "CCAvg": ccavg,
        "Education": education,
        "Mortgage": mortgage,
        "Securities.Account": securities_account,
        "CD.Account": cd_account,
        "Online": online,
        "CreditCard": creditcard
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display input
st.subheader("Entered Customer Details")
st.write(input_df)

# Predict button
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Loan will likely be APPROVED (Confidence: {prediction_proba:.2%})")
    else:
        st.error(f"❌ Loan will likely be REJECTED (Confidence: {1-prediction_proba:.2%})")