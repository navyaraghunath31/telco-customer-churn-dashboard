import streamlit as st
import pandas as pd
import numpy as np

def show_predict(X, label_encoders, scaler, model, y, df):
    st.header('🔮 **Predict Customer Churn**')
    st.sidebar.header('Enter Customer Details')
    user_input = {}
    for col in X.columns:
        if X[col].dtype == 'int64' or X[col].dtype == 'float64':
            min_val = float(X[col].min())
            max_val = float(X[col].max())
            mean_val = float(X[col].mean())
            user_input[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)
        else:
            options = sorted(X[col].unique())
            user_input[col] = st.sidebar.selectbox(col, options=options)
    input_df = pd.DataFrame([user_input])
    input_encoded = input_df.copy()
    for col in input_encoded.select_dtypes(include=['object']).columns:
        le = label_encoders[col]
        input_encoded[col] = le.transform(input_encoded[col])
    input_scaled = scaler.transform(input_encoded)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    st.subheader('**Prediction Results**')
    predict_clicked = st.sidebar.button('Predict Churn')
    if predict_clicked:
        churn_rate = y.mean()
        if churn_rate < 0.1:
            st.warning('Warning: The dataset is highly imbalanced. The model may be biased toward predicting customers will stay.')
        result_color = '#C8E6C9' if pred == 0 else '#FFCDD2'
        result_icon = '✅' if pred == 0 else '⚠️'
        result_text = 'This customer is likely to stay.' if pred == 0 else 'This customer is at risk of churning!'
        prob_color = '#388E3C' if pred == 0 else '#B71C1C'
        st.markdown(f'''
        <div style="background-color:{result_color};padding:24px 18px 18px 18px;border-radius:14px;margin-bottom:18px;text-align:center;box-shadow:0 2px 8px #eee;">
            <h2 style="color:{prob_color};font-weight:bold;margin-bottom:10px;">{result_icon} {result_text}</h2>
            <div style="font-size:22px;font-weight:bold;color:{prob_color};margin-bottom:8px;">Churn Probability</div>
            <div style="font-size:32px;font-weight:bold;color:{prob_color};margin-bottom:18px;">{prob:.2f}</div>
            <div style="background-color:#FFF3E0;padding:10px 18px;border-radius:10px;margin-bottom:10px;text-align:center;font-size:16px;color:#FF9800;font-weight:bold;">
                <span style="font-size:20px;">🏆</span> <b>Top drivers of churn:</b><br>
                <span style="font-size:16px;">MonthlyCharges, tenure, TotalCharges, Contract, PaymentMethod</span>
            </div>
            <div style="background-color:#e3f2fd;padding:8px;border-radius:8px;margin-top:8px;text-align:center;font-size:15px;color:#1976d2;font-weight:bold;">🌟 Use these insights to guide retention strategies and marketing!</div>
        </div>
        ''', unsafe_allow_html=True)
        with st.expander('📊 Compare Input to Population', expanded=False):
            top_features = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract', 'PaymentMethod']
            num_features = [f for f in top_features if f in input_df.columns and input_df[f].dtype in [np.int64, np.float64]]
            if num_features:
                input_vals = input_df[num_features].iloc[0]
                avg_vals = df[num_features].mean()
                chart_df = pd.DataFrame({'Customer Input': input_vals, 'Population Avg': avg_vals})
                chart_df.plot(kind='bar', figsize=(7,4), color=['#4F8BF9','#FF9800'])
                import matplotlib.pyplot as plt
                plt.title('Customer Input vs. Population Average', color='#1976d2')
                plt.ylabel('Value')
                st.pyplot(plt.gcf())
            else:
                st.info('No numeric top driver features available for visualization.')
            st.markdown('<div style="background-color:#FFF3E0;padding:8px;border-radius:8px;margin-top:8px;text-align:center;font-size:14px;color:#FF9800;font-weight:bold;">🔎 See how your scenario compares to typical customers for key churn drivers.</div>', unsafe_allow_html=True)
