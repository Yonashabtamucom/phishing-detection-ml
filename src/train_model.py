import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

from preprocessing import preprocess_data
from features import create_features
from evaluate import evaluate_model

# Load dataset (LOCAL CSV IS FINE)
dataset_path = r"C:\Users\Student\Downloads\PhishingData\PhishingData.csv"
df = pd.read_csv(dataset_path)

# Preprocess
df_clean = preprocess_data(df)

# Features and label
X, y = create_features(df_clean, target_column="Result")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model

model_path = r"C:\Users\Student\Desktop\phishing_model.pkl"
joblib.dump(model, model_path)




# Evaluate
evaluate_model(model, X_test, y_test)
