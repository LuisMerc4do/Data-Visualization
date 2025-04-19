import numpy as np

class WeightedNearestNeighbor:
    # References:
    # https://scikit-learn.org/stable/modules/neighbors.html
    # https://medium.com/@lakshmiteja.ip/understanding-weighted-k-nearest-neighbors-k-nn-algorithm
    # https://www.youtube.com/watch?v=xXLTYsHfWzg&ab_channel=CampusX

    def __init__(self):
        # Initialize empty placeholders for training data and labels
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training dataset and corresponding labels.

        Parameters:
        X : numpy array of shape (n_samples, 5) — training features in R^5
        y : numpy array of shape (n_samples,) — binary class labels {0, 1}
        """
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X):
        """
        Predict labels for new data using weighted nearest neighbor approach.

        Parameters:
        X : numpy array of shape (n_samples, 5) — test samples to classify

        Returns:
        y_pred : numpy array of predicted labels
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("You must call fit() before predict().")

        # Placeholder for predictions
        y_pred = np.zeros(X.shape[0])

        # Iterate through each test sample
        for i, test_point in enumerate(X):
            # Step 1: Compute distances to all training samples (Euclidean distance)
            distances = np.sqrt(np.sum((self.X_train - test_point) ** 2, axis=1))

            # Step 2: If test point exactly matches a training point
            if np.any(distances == 0):
                exact_match_index = np.argmin(distances)
                y_pred[i] = self.y_train[exact_match_index]
                continue

            # Step 3: Compute inverse distances to weight nearby points more heavily
            with np.errstate(divide='ignore'):  # Prevent division warnings
                inv_distances = 1.0 / distances

            # Step 4: Handle infinite weights caused by zero distance
            inv_distances[np.isinf(inv_distances)] = 1e10

            # Step 5: Normalize weights (ensure they sum to 1)
            weights = inv_distances / np.sum(inv_distances)

            # Step 6: Aggregate weights by class
            class_0_score = np.sum(weights[self.y_train == 0])
            class_1_score = np.sum(weights[self.y_train == 1])

            # Step 7: Assign class with greater total weight (higher influence)
            y_pred[i] = 0 if class_0_score >= class_1_score else 1

        return y_pred

# ------------------
# Demonstration example with interpretable result
if __name__ == "__main__":
    # Class 1 is located very close to test sample
    X_train = np.array([
        [0, 0, 0, 0, 0],    # Class 1 — close
        [5, 5, 5, 5, 5],    # Class 0 — far
        [6, 6, 6, 6, 6],    # Class 0 — far
        [7, 7, 7, 7, 7]     # Class 0 — far
    ])
    y_train = np.array([1, 0, 0, 0])

    # Test point very close to the Class 1 sample
    X_test = np.array([[0.1, 0.1, 0.1, 0.1, 0.1]])

    # Instantiate classifier, train, and predict
    clf = WeightedNearestNeighbor().fit(X_train, y_train)
    prediction = clf.predict(X_test)

    # Output prediction
    print("Test point:", X_test[0])
    print("Predicted class:", int(prediction[0]))
