import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("synthetic_stress_heart_rate.csv")  # Ensure this file is in your working directory

# Encode stress levels50
df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Features and labels
X = df[['Heart_Rate']]  # Add more features if available
y = df['Stress_Level']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "stress_prediction_model.pkl")

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Function to predict stress level for a given heart rate
def predict_stress(heart_rate):
    stress_mapping = {0: "Low", 1: "Medium", 2: "High"}
    stress_level = model.predict([[heart_rate]])[0]  # Predict using pre-loaded model
    return stress_mapping[stress_level]

# Example: Predict stress level for a random heart rate (e.g., 85 BPM)
random_heart_rate = int(input("Enter heart rate: "))
predicted_stress = predict_stress(random_heart_rate)
print("Predicted Stress Level:", predicted_stress)
