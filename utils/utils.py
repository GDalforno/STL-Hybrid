import numpy as np


class Queue:
    def __init__(self, data:np.ndarray):
        self.__data = data.tolist()

    def enqueue(self, x:float)->None:
        self.__data.append(x)

    def dequeue(self)->float:
        return self.__data.pop(0)

    def as_array(self)->np.ndarray:
        return np.array(self.__data)

    
def preprocess(x:np.ndarray, window_length:int, detrend:bool=True)->tuple:
    X, y = [], []
    window_length = window_length+1 if detrend else window_length
    n_samples = len(x) - window_length
    for i in range(n_samples):
        if detrend:
            y0 = x[i+window_length-1]
            X_, y_ = x[i:i+window_length-1]-y0, x[i+window_length]-y0
        else:
            X_, y_ = x[i:i+window_length], x[i+window_length]
        X.append(X_)
        y.append(y_)

    return (np.array(X), np.array(y))

