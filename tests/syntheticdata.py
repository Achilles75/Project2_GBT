import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=200, random_seed=42):
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, 2)

    # Nonlinear decision boundary: XOR-like logic
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    return X, y

# Plot the data to visualize class distribution
def plot_data(X, y, title="Synthetic Classification Data"):
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=60)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    X, y = generate_synthetic_data(n_samples=300)
    plot_data(X, y)
