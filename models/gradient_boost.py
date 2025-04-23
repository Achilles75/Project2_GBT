
import numpy as np
from models.regression_tree import RegressionTree

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.init_pred = 0.0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Initialize with log odds
        y = np.array(y)
        self.init_pred = np.log(np.mean(y) / (1 - np.mean(y)))
        pred = np.full(y.shape, self.init_pred)

        for _ in range(self.n_estimators):
            residual = y - self._sigmoid(pred)
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            update = tree.predict(X)
            pred += self.learning_rate * update
            self.models.append(tree)

    def predict_proba(self, X):
        pred = np.full((X.shape[0],), self.init_pred)
        for tree in self.models:
            pred += self.learning_rate * tree.predict(X)
        prob = self._sigmoid(pred)
        return np.vstack([1 - prob, prob]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
