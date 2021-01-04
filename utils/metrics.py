import numpy as np
import pandas as pd


def pocid(y_test:np.ndarray, y_pred:np.ndarray)->float:
    d = 0
    h = len(y_test)
    for i in range(1, h):
        d += 1 if (y_test[i]-y_test[i-1])*(y_pred[i]-y_pred[i-1]) > 0 else 0
    return d/h

def multi_criteria(model_metrics:pd.Series)->float:
    a, b, c = model_metrics
    c = 1-c
    S = np.sin(2*np.pi/3)
    return S/2*(a*b + a*c + b*c)

