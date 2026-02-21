import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("🛒 Customer Cluster Prediction using KMeans")

# Sample dataset
data = {
    "Age": [22, 25, 47, 52, 46, 56, 23, 44, 36, 29],
    "Income": [15, 16, 59, 60, 58, 62, 18, 55, 40, 30],
    "SpendingScore": [39, 81, 6, 77, 40, 76, 94, 3, 50, 60]
}

df = pd.DataFrame(data)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)

st.subheader("Enter Customer Details")

age = st.number_input("Age", min_value=10, max_value=100, value=25)
income = st.number_input("Annual Income", min_value=1, max_value=100, value=30)
score = st.number_input("Spending Score", min_value=1, max_value=100, value=50)

if st.button("Predict Customer Cluster"):
    prediction = kmeans.predict([[age, income, score]])
    st.success(f"Customer belongs to Cluster: {prediction[0]}")