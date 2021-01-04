import numpy as np
from sklearn.metrics import mean_absolute_error


def pocid(y_test:np.ndarray, y_pred:np.ndarray)->float:
    d = 0
    h = len(y_test)
    for i in range(1, h):
        d += 1 if (y_test[i]-y_test[i-1])*(y_pred[i]-y_pred[i-1]) > 0 else 0
    return d/h

