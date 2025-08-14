from sklearn.ensemble import RandomForestClassifier

def train_model(X_scaled, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled, y)
    return model
