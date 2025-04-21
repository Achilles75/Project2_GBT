import sys
import os

# Add the project root to sys.path so we can import from model/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from model.gradientboostingclassifier import GradientBoostingClassifier
from tests.syntheticdata import generate_synthetic_data, plot_data
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

def test_boosting_on_synthetic_data():
    # Load data
    X, y = generate_synthetic_data(n_samples=300)

    # Train model
    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1)
    clf.fit(X, y)

    # Predict
    y_pred = clf.predict(X)

    # Evaluate
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)

    assert acc > 0.85, "Expected accuracy > 85% on synthetic data"

    # Optional visualization
    plot_decision_boundary(clf, X, y)

def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='bwr', alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=60)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_boosting_on_synthetic_data()
