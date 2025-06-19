import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

st.title("💳 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("📂 Upload a transaction CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Input Data:", data.head())

    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    scaler = StandardScaler()
    data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

    model = xgb.XGBClassifier()
    model.load_model("xgboost_model.json")

    predictions = model.predict(data)
    data['Fraud Prediction'] = predictions

    st.write("✅ Predictions:", data[['Fraud Prediction']])
