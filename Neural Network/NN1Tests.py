import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
training_data = pd.read_csv('Neural Network\AtRiskStudentsTraining.csv')
test_data = pd.read_csv('Neural Network\AtRiskStudentsTest.csv')

# Separate features and target
X_train = training_data.iloc[:, 0:4].values
y_train = training_data.iloc[:, 4].values
X_test = test_data.iloc[:, 0:4].values
y_test = test_data.iloc[:, 4].values

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define various network architectures to test
architectures = [
    ((5,), "1 hidden layer, 5 neurons"),
    ((10,), "1 hidden layer, 10 neurons"),
    ((20,), "1 hidden layer, 20 neurons"),
    ((5, 5), "2 hidden layers, 5 neurons each"),
    ((10, 10), "2 hidden layers, 10 neurons each"),
    ((20, 10), "2 hidden layers, 20 and 10 neurons"),
    ((10, 10, 5), "3 hidden layers, 10, 10, and 5 neurons"),
    ((20, 15, 10), "3 hidden layers, 20, 15, and 10 neurons"),
    ((30, 20, 10), "3 hidden layers, 30, 20, and 10 neurons"),
    ((15, 12, 10, 5), "4 hidden layers, 15, 12, 10, and 5 neurons")
]

# Dictionary to store results
results = []

# Test each architecture
for hidden_layer_sizes, description in architectures:
    print(f"\nTraining model with {description}...")
    
    # Create and train model
    nn_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=1000,
        random_state=42
    )
    
    nn_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate log loss
    error = log_loss(y_test, y_pred_proba)
    
    # Store results
    results.append({
        'Architecture': description,
        'Hidden Layers': len(hidden_layer_sizes),
        'Neurons': hidden_layer_sizes,
        'Error': error
    })
    
    print(f"Error (Log Loss): {error:.4f}")

# Convert results to DataFrame for easy display
results_df = pd.DataFrame(results)
print("\nResults Table:")
print(results_df[['Architecture', 'Hidden Layers', 'Error']].to_string(index=False))

# Create and save a visualization of results
plt.figure(figsize=(12, 6))
sns.barplot(x='Architecture', y='Error', data=results_df)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('nn_architecture_comparison.png')

# Save results to CSV
results_df.to_csv('Neural Network\ nn_architecture_results.csv', index=False)

print("\nAnalysis complete. Results saved to 'nn_architecture_results.csv'")