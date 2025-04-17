import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Function to generate synthetic data (you can replace this with your actual dataset)
def generate_synthetic_data(num_samples=10000):
    heart_rate = np.random.normal(75, 10, num_samples)  
    hrv = np.random.normal(50, 15, num_samples)
    time_in_elevated_zone = np.random.uniform(0, 1, num_samples) * (heart_rate > 90)
    resting_heart_rate = np.random.uniform(60, 70, num_samples)
    stress_level = np.array(["Low" if hr < 70 else "Medium" if hr < 90 else "High" for hr in heart_rate])
    
    df = pd.DataFrame({
        'heart_rate': heart_rate,
        'hrv': hrv,
        'time_in_elevated_zone': time_in_elevated_zone,
        'resting_heart_rate': resting_heart_rate,
        'stress_level': stress_level
    })
    return df

# Generate synthetic data
df = generate_synthetic_data(1000)

# Prepare the features (X) and target (y)
X = df.drop(columns=['stress_level'])
y = df['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features and transform them
X_scaled = scaler.fit_transform(X)

# Save the scaler using joblib
joblib.dump(scaler, 'scaler.pkl')

# Show the first few rows of the scaled data
scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(scaled_df.head())

