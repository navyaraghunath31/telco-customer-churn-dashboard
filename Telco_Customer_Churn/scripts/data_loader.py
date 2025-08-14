import pandas as pd
import numpy as np
import os

def load_data():
    # Get the absolute path to the data file
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'data', 'Telco-Customer-Churn.csv')
    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df = df.replace(' ', np.nan)
    df = df.dropna()
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df



