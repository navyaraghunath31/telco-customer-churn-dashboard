import streamlit as st
from scripts.data_loader import load_data
from scripts.preprocessing import preprocess_data
from scripts.model import train_model
from scripts.dashboard import show_dashboard
from scripts.predict import show_predict

# Custom CSS
st.markdown('''<style>
    .main {background-color: #f7f9fa;}
    .block-container {padding-top: 2rem;}
    .stMetric {background-color: #e3f2fd; border-radius: 10px;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
.streamlit-tabs [data-baseweb="tab"] {font-weight:700;font-size:20px;}
</style>''', unsafe_allow_html=True)

st.set_page_config(page_title="Telco Customer Churn Dashboard", page_icon="📊", layout="wide")

# Sidebar branding and information
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
st.sidebar.title("Telco Churn Project")
st.sidebar.markdown("""
**<span style='font-weight:bold; color:#D20103;'>This project is by Navya R.</span>**

**<span style='font-weight:bold'>Business Context:</span>**
Churn prediction helps telecom companies identify customers likely to leave, enabling targeted retention strategies and reducing revenue loss.

**<span style='font-weight:bold'>How to Use:</span>**
- Explore the <b>Dashboard</b> tab for insights and visualizations
- Use the <b>Predict Churn</b> tab to test scenarios and get predictions
""", unsafe_allow_html=True)

st.sidebar.markdown("""
---

**<span style='font-weight:bold'>Contact:</span>**
:email: navyaraghunath31@gmail.com
""", unsafe_allow_html=True)

# Load and preprocess data
df = load_data()
X, y, X_scaled, label_encoders, scaler = preprocess_data(df)
model = train_model(X_scaled, y)

# Add prominent heading and subtitle to main page
st.markdown('''
<div style="background: linear-gradient(90deg, #1976d2 0%, #64b5f6 100%); padding: 28px 10px 18px 10px; border-radius: 18px; margin-bottom: 28px; text-align: center; box-shadow: 0 2px 12px #e3f2fd;">
    <h1 style="color: #fff; font-size: 38px; font-weight: bold; margin-bottom: 10px; letter-spacing: 1px;">Telco Customer Churn Dashboard & Prediction</h1>
    <div style="font-size: 20px; color: #fff; font-weight: 500; margin-bottom: 4px;">Explore churn insights and make predictions.</div>
    <div style="font-size: 16px; color: #e3f2fd; font-weight: 500;">Use the tabs below to switch between dashboard and prediction!</div>
</div>
''', unsafe_allow_html=True)

# Tab names (plain text)
tab_names = ["Dashboard", "Predict Churn"]
selected_tab = st.tabs(tab_names)

with selected_tab[0]:
    show_dashboard(df, X, model)

with selected_tab[1]:
    show_predict(X, label_encoders, scaler, model, y, df)

