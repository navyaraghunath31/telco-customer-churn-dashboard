import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def show_dashboard(df, X, model):
    st.markdown('<div style="background-color:#e3f2fd;padding:18px 10px 10px 10px;border-radius:12px;margin-bottom:20px;text-align:center;">'
                '<h2 style="color:#1976d2;font-weight:bold;margin-bottom:10px;">🌈 Churn Data Visualizations</h2>'
                '<div style="display:flex;justify-content:space-around;flex-wrap:wrap;">'
                f'<div style="margin:10px 20px;min-width:120px;">'
                '<span style="font-size:22px;font-weight:bold;color:#B71C1C;">💔 Churn Rate</span><br>'
                f'<span style="font-size:28px;font-weight:bold;color:#B71C1C;">{df["Churn"].mean()*100:.2f}%</span>'
                '</div>'
                f'<div style="margin:10px 20px;min-width:120px;">'
                '<span style="font-size:22px;font-weight:bold;color:#1976d2;">👥 Total Customers</span><br>'
                f'<span style="font-size:28px;font-weight:bold;color:#1976d2;">{len(df):,}</span>'
                '</div>'
                f'<div style="margin:10px 20px;min-width:120px;">'
                '<span style="font-size:22px;font-weight:bold;color:#388E3C;">💸 Avg Monthly Charges</span><br>'
                f'<span style="font-size:28px;font-weight:bold;color:#388E3C;">${df["MonthlyCharges"].mean():.2f}</span>'
                '</div>'
                f'<div style="margin:10px 20px;min-width:120px;">'
                '<span style="font-size:22px;font-weight:bold;color:#FF9800;">⏳ Avg Tenure (months)</span><br>'
                f'<span style="font-size:28px;font-weight:bold;color:#FF9800;">{df["tenure"].mean():.1f}</span>'
                '</div>'
                '</div></div>', unsafe_allow_html=True)
    with st.expander("🎨 **Churn Distribution**", expanded=True):
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='Churn', data=df, ax=ax, palette='pastel', hue='Churn', legend=False)
        ax.set_title('Churn Distribution', color='#F44336')
        st.pyplot(fig)
    with st.expander("📈 **Feature Distributions by Churn**", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols.remove('Churn')
        selected_num = st.selectbox('Select a numerical feature:', num_cols)
        fig, ax = plt.subplots(figsize=(6,4))
        sns.violinplot(x='Churn', y=selected_num, data=df, ax=ax, palette='coolwarm', hue='Churn', legend=False)
        ax.set_title(f'{selected_num} by Churn', color='#4F8BF9')
        st.pyplot(fig)
    with st.expander("📝 **Filter by Contract Type**", expanded=False):
        contract_types = sorted(df['Contract'].unique())
        selected_contract = st.selectbox('Contract Type:', contract_types)
        filtered_df = df[df['Contract'] == selected_contract]
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x='Churn', data=filtered_df, ax=ax, palette='Set1', hue='Churn', legend=False)
        ax.set_title(f'Churn by Contract: {selected_contract}', color='#FF9800')
        st.pyplot(fig)
    with st.expander("🔥 **Correlation Heatmap**", expanded=False):
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Feature Correlation Heatmap', color='#1976d2')
        st.pyplot(fig)
    with st.expander("🏆 **Top Feature Importances**", expanded=False):
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots(figsize=(8,6))
        feat_importances.nlargest(10).plot(kind='barh', ax=ax, color='#FF9800')
        ax.set_title('Top 10 Feature Importances', color='#388E3C')
        st.pyplot(fig)
    st.markdown('<div style="background-color:#FFF3E0;padding:10px;border-radius:10px;margin-top:10px;text-align:center;font-size:15px;color:#FF9800;font-weight:bold;">🌟 Tip: Use the filters and expanders to explore churn patterns and drivers interactively.</div>', unsafe_allow_html=True)
