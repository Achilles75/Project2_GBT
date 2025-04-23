
import numpy as np
import pandas as pd
from models.gradient_boost import GradientBoostingClassifier
from utils.metrics import *
from utils.visualize_results import plot_confusion_matrix, plot_roc_curve

def load_data(filepath='data/ibm_attrition.csv'):
    data = pd.read_csv(filepath)
    data = data.dropna()
    # Attrition is already numeric (0 and 1), so no mapping needed
    y = data['Attrition'].values
    X = data.select_dtypes(include=[np.number]).drop('EmployeeNumber', axis=1, errors='ignore').values
    return X, y

def main():
    X, y = load_data()
    np.random.seed(42)

    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]

    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)

    # Ensure at least 10 class 1 examples in training
    train_pos = pos_indices[:10]
    test_pos = pos_indices[10:]

    split_neg = int(0.8 * len(neg_indices))
    train_neg = neg_indices[:split_neg]
    test_neg = neg_indices[split_neg:]

    train_indices = np.concatenate((train_pos, train_neg))
    test_indices = np.concatenate((test_pos, test_neg))

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    # Manual oversampling of class 1
    X_train_class0 = X_train[y_train == 0]
    y_train_class0 = y_train[y_train == 0]
    X_train_class1 = X_train[y_train == 1]
    y_train_class1 = y_train[y_train == 1]

    if len(y_train_class1) == 0:
        raise ValueError("No class 1 samples available in training set.")

    repeat_factor = max(1, len(y_train_class0) // len(y_train_class1))
    X_class1_upsampled = np.repeat(X_train_class1, repeat_factor, axis=0)
    y_class1_upsampled = np.repeat(y_train_class1, repeat_factor)

    X_train_balanced = np.vstack((X_train_class0, X_class1_upsampled))
    y_train_balanced = np.concatenate((y_train_class0, y_class1_upsampled))
    shuffle_idx = np.random.permutation(len(y_train_balanced))
    X_train_balanced = X_train_balanced[shuffle_idx]
    y_train_balanced = y_train_balanced[shuffle_idx]

    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_proba):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_proba):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_proba):.4f}")
    print(f"R² Score: {r2_score(y_test, y_proba):.4f}")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)

if __name__ == "__main__":
    main()
