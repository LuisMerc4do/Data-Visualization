import numpy as np

class WeightedNearestNeighbor:
    #ref. https://scikit-learn.org/stable/modules/neighbors.html
    #ref. https://medium.com/@lakshmiteja.ip/understanding-weighted-k-nearest-neighbors-k-nn-algorithm
    #ref. https://www.youtube.com/watch?v=xXLTYsHfWzg&ab_channel=CampusX 
    def __init__(self):
        # Initialize placeholders for training data
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Store training data and labels.
        
        Parameters:
        X : numpy array of shape (n_samples, 5) — feature vectors in R^5.
        y : numpy array of shape (n_samples,) — binary class labels in {0, 1}.
        """
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """
        Predict the class for each test sample using weighted nearest neighbors.
        
        Parameters:
        X : numpy array of shape (n_samples, 5) — test points to classify.
        
        Returns:
        y_pred : numpy array of shape (n_samples,) — predicted class labels.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must call fit() before predict().")
        
        # Initialize output array to store predictions
        y_pred = np.zeros(X.shape[0])
        
        # Loop through each test sample
        for i, test_point in enumerate(X):
            # Step 1: Compute Euclidean distances between test point and all training points
            distances = np.sqrt(np.sum((self.X_train - test_point)**2, axis=1))

            # Step 2: Handle exact matches (distance = 0) to avoid division by zero
            if np.any(distances == 0):
                # If the test point exactly matches a training point, use that label directly
                exact_match_index = np.argmin(distances)
                y_pred[i] = self.y_train[exact_match_index]
                continue

            # Step 3: Compute inverse distances so closer points have higher influence
            # Use numpy's error handling to suppress warnings for divide-by-zero
            with np.errstate(divide='ignore'):
                inv_distances = 1.0 / distances

            # Step 4: Set infinite weights (due to 0 distance) to a very large number
            inv_distances[np.isinf(inv_distances)] = 1e10

            # Step 5: Normalize weights to sum to 1
            weights = inv_distances / np.sum(inv_distances)

            # Step 6: Aggregate the weights for each class separately
            class_0_score = np.sum(weights[self.y_train == 0])  # Total weight of class 0
            class_1_score = np.sum(weights[self.y_train == 1])  # Total weight of class 1

            # Step 7: Predict class with the higher total weight (more influence)
            y_pred[i] = 0 if class_0_score >= class_1_score else 1
        
        return y_pred


# Demonstration example
if __name__ == "__main__":
    # Training data in R^5
    # Class 1 is very close to test point, class 0 samples are farther away
    X_train = np.array([
        [0, 0, 0, 0, 0],    # Class 1 (close to test point)
        [5, 5, 5, 5, 5],    # Class 0 (far)
        [6, 6, 6, 6, 6],    # Class 0 (far)
        [7, 7, 7, 7, 7]     # Class 0 (far)
    ])
    y_train = np.array([1, 0, 0, 0])

    # Test point very close to the class 1 sample
    X_test = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])

    # Create classifier, train on training data, and predict
    clf = WeightedNearestNeighbor().fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # Output prediction
    print("Test point:", X_test[0])
    print("Predicted class:", int(prediction[0]))
