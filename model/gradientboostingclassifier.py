import numpy as np

class DecisionStump:
    """A simple decision stump for binary classification"""
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        # Greedy search for best threshold and feature
        n_samples, n_features = X.shape
        best_loss = float('inf')

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.any(left_mask) and np.any(right_mask):
                    left_val = np.mean(residuals[left_mask])
                    right_val = np.mean(residuals[right_mask])

                    preds = np.where(left_mask, left_val, right_val)
                    loss = np.mean((residuals - preds) ** 2)

                    if loss < best_loss:
                        self.feature_index = feature
                        self.threshold = threshold
                        self.left_value = left_val
                        self.right_value = right_val
                        best_loss = loss

    def predict(self, X):
        return np.where(X[:, self.feature_index] <= self.threshold,
                        self.left_value, self.right_value)


class GradientBoostingClassifier:
    def __init__(self, n_estimators=500, learning_rate=0.05):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        y = y * 2 - 1  # Convert {0, 1} → {-1, 1}
        self.initial_pred = 0.5 * np.log((1 + np.mean(y)) / (1 - np.mean(y)))  # Initial log odds
        pred = np.full(y.shape, self.initial_pred)

        for _ in range(self.n_estimators):
            residuals = y / (1 + np.exp(y * pred))  # Negative gradient of log loss
            stump = DecisionStump()
            stump.fit(X, residuals)
            update = stump.predict(X)
            pred += self.learning_rate * update
            self.trees.append(stump)

    def predict(self, X):
        pred = np.full(X.shape[0], self.initial_pred)
        for stump in self.trees:
            pred += self.learning_rate * stump.predict(X)
        return (pred > 0).astype(int)
