import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, log_loss

# Load test data
test_data = pd.read_csv('AtRiskStudentsTest.csv')

# Separate features and target
X_test = test_data.iloc[:, 0:4].values  # GPA, attendance, duration, language
y_test = test_data.iloc[:, 4].values    # at-risk label

# Load the trained model and scaler
nn_model = joblib.load('nn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Standardize test features
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = nn_model.predict(X_test_scaled)
y_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
cm = confusion_matrix(y_test, y_pred)

# Print results
print(f"Model Performance Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Log Loss: {logloss:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Our primary error function will be the log loss (cross-entropy)
print(f"\nSelected Error Function: Log Loss (Cross-Entropy)")
print(f"Error Value: {logloss:.4f}")