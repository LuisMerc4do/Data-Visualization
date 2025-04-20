import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.metrics import mean_squared_error, log_loss

# Load test data
test_data = pd.read_csv('Neural Network\AtRiskStudentsTest.csv')

# Separate features and target
X_test = test_data.iloc[:, 0:4].values  # GPA, attendance, duration, language
y_test = test_data.iloc[:, 4].values    # at-risk label

# Load the trained model and scaler
nn_model = joblib.load('Neural Network\ nn1_model.pkl')
scaler = joblib.load('Neural Network\ nn1_scaler.pkl')

# Standardize test features
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = nn_model.predict(X_test_scaled)
y_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics, more types of metrics could be implemented
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)

# Print results
print(f"Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Log Loss: {logloss:.4f}")

# Our primary error function will be the log loss
print(f"\nLog Loss")
print(f"Error Value: {logloss:.4f}")