import numpy as np
from sklearn.metrics import auc


def is_key_exists(k, item):
    return (k in item) and (item[k] is not None)


def calculate_auc(mean_accuracy_by_step: np.ndarray) -> float:
    x = np.array([0.0, 0.33, 0.66, 1.0])
    return auc(x = x, y = mean_accuracy_by_step)
