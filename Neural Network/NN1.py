import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
#References
#https://medium.com/@michaeldelsole/a-single-layer-artificial-neural-network-in-20-lines-of-python-ae34b47e5fef
#https://how.dev/answers/implement-neural-network-for-classification-using-scikit-learn 
#https://medium.com/@eshan.k.iyer/how-to-use-a-neural-network-and-scikit-learn-a-practical-implementation-61d890b133d7 

# Load training data
data = pd.read_csv('Neural Network\\AtRiskStudentsTraining.csv')
X_train = data[['GPA', 'attendance', 'duration', 'language']]
y_train = data['at-risk']

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Define and train NN
model = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler for test and validation
joblib.dump(model, 'Neural Network\\NN1_model.pkl')
joblib.dump(scaler, 'Neural Network\\NN1_scaler.pkl')
