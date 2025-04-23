
import numpy as np

class RegressionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = np.zeros(0)

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == 0:
            self.feature_importances_ = np.zeros(X.shape[1])

        if depth >= self.max_depth or len(set(y)) == 1:
            return np.mean(y)

        best_feature = None
        best_split = None
        best_loss = float('inf')

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for t in thresholds:
                left_mask = X[:, feature_idx] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_loss = np.var(y[left_mask]) * len(y[left_mask])
                right_loss = np.var(y[right_mask]) * len(y[right_mask])
                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_feature = feature_idx
                    best_split = t

        if best_feature is None:
            return np.mean(y)

        self.feature_importances_[best_feature] += 1
        left_mask = X[:, best_feature] <= best_split
        right_mask = ~left_mask

        return {
            'feature': best_feature,
            'split': best_split,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def _predict_sample(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['split']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])
