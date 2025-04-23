import numpy as np
import pandas as pd
import os

def generate_data(n_samples=200, random_state=42):
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    data = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
    data['label'] = y
    return data

def save_to_csv(data, filename='data/synthetic_data.csv'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data.to_csv(filename, index=False)

if __name__ == "__main__":
    data = generate_data()
    save_to_csv(data)
    print("Synthetic data saved to 'data/synthetic_data.csv'")
