import pandas as pd
from data import path, max_p
from sktime.forecasting.model_selection import temporal_train_test_split


def load_dataset(index:int, test_size:float=0.05)->tuple:
    assert index<len(path), "Index out of range"
    
    data = pd.read_csv(path[index], header=None).values.reshape(-1)
    (x, xx) = temporal_train_test_split(data, test_size=test_size)
    
    return (x, xx, max_p[index])
