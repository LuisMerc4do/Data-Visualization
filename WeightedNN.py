# WeightedNN.py
import numpy as np

def weighted_nn_classifier(train_X, train_y, test_X):
    """
    Weighted nearest neighbour classifier.

    Args:
        train_X (np.ndarray): shape (n_samples, 5), training features
        train_y (np.ndarray): shape (n_samples,), class labels (0 or 1)
        test_X (np.ndarray): shape (m_samples, 5), test features

    Returns:
        np.ndarray: shape (m_samples,), predicted labels (0 or 1)
    """
    predictions = []

    for test_point in test_X:
        distances = np.linalg.norm(train_X - test_point, axis=1)
        distances = np.where(distances == 0, 1e-10, distances)  # Avoid division by zero

        weights = distances / np.sum(distances)

        # Aggregate weighted vote
        vote_0 = np.sum(weights[train_y == 0])
        vote_1 = np.sum(weights[train_y == 1])

        predicted_label = 0 if vote_0 > vote_1 else 1
        predictions.append(predicted_label)

    return np.array(predictions)
