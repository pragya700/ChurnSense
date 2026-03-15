import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction App")

# Load model
model = joblib.load("churn_model.pkl")

# Inputs
tenure = st.number_input("Tenure Months", 0, 72)
monthly = st.number_input("Monthly Charges", 0.0, 200.0)
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])

# Create input row
input_df = pd.DataFrame({
    "Tenure Months":[tenure],
    "Monthly Charges":[monthly],
    "Contract":[contract]
})

# Encode like training
input_df = pd.get_dummies(input_df)

# Match training columns
model_cols = model.feature_names_in_
input_df = input_df.reindex(columns=model_cols, fill_value=0)

# Predict button
if st.button("Predict"):
    pred = model.predict(input_df)

    if pred[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")