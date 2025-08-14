import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv('Telco-Customer-Churn/data/Telco-Customer-Churn.csv')
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df = df.replace(' ', np.nan)
    df = df.dropna()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

