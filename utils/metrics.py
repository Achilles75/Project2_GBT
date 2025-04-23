
import numpy as np

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall + 1e-8)

def roc_auc_score(y_true, y_pred_prob):
    thresholds = np.linspace(0, 1, 100)
    tpr, fpr = [], []
    for threshold in thresholds:
        y_pred = (y_pred_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr.append(tp / (tp + fn + 1e-8))
        fpr.append(fp / (fp + tn + 1e-8))
    return np.trapz(tpr, fpr)

def log_loss(y_true, y_proba, eps=1e-15):
    y_proba = np.clip(y_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba))

def mean_squared_error(y_true, y_proba):
    return np.mean((y_true - y_proba) ** 2)

def mean_absolute_error(y_true, y_proba):
    return np.mean(np.abs(y_true - y_proba))

def r2_score(y_true, y_proba):
    mean_y = np.mean(y_true)
    ss_tot = np.sum((y_true - mean_y) ** 2)
    ss_res = np.sum((y_true - y_proba) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
