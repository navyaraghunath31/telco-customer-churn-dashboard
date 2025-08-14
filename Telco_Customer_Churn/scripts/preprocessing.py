import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    scaler = StandardScaler()
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
        label_encoders[col] = le
    X_scaled = scaler.fit_transform(X_encoded)
    return X, y, X_scaled, label_encoders, scaler
