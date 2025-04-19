import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load training data
training_data = pd.read_csv('AtRiskStudentsTraining.csv')

# Separate features and target
X_train = training_data.iloc[:, 0:4].values  # GPA, attendance, duration, language
y_train = training_data.iloc[:, 4].values    # at-risk label

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create and train the neural network with 1 hidden layer
# Hidden layer with 10 neurons, using ReLU activation function
nn_model = MLPClassifier(
    hidden_layer_sizes=(10,),  # One hidden layer with 10 neurons
    activation='relu',         # ReLU activation function
    solver='adam',             # Adam optimizer
    alpha=0.0001,              # L2 regularization parameter
    max_iter=1000,             # Maximum number of iterations
    random_state=42,           # For reproducibility
    verbose=True
)

# Train the model
nn_model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(nn_model, 'nn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Neural Network trained and saved successfully.")