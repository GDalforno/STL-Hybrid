import numpy as np
from sklearn.base import RegressorMixin
from statsmodels.tsa.seasonal import STL


class Queue:
    def __init__(self, data:np.ndarray):
        self.__data = data.tolist()

    def enqueue(self, x:float)->None:
        self.__data.append(x)

    def dequeue(self)->float:
        return self.__data.pop(0)

    def as_array(self)->np.ndarray:
        return np.array(self.__data)


class SeasonalNaive:
    def __init__(self, period:int):
        self.__period = period
        self.__data = None
    
    def fit(self, data:np.ndarray)->None:
        self.__data = data[-self.__period:]
        
    def predict(self, h:np.int)->np.ndarray:
        y = np.tile(self.__data, h//self.__period+1)
        return y[:h]


def preprocess(x:np.ndarray, window_length:int)->tuple:
    X, y = [], []
    n_samples = len(x) - window_length
    for i in range(n_samples):
        y0 = x[i+window_length-1]
        X.append(x[i:i+window_length]-y0)
        y.append(x[i+window_length]-y0)

    return (np.array(X), np.array(y))


class TimeSeriesForecaster:
    def __init__(
            self, model:RegressorMixin, window_length:int, max_p:int=None):
        self.__window_length = window_length  
        self.__max_p = max_p 
        self.__model = model
        self.__snaivef = None 
        self.__stl = None 
        self.__x = None
        self.__m = None
        self.__queue = None

    def fit(self, x:np.ndarray)->None:
        self.__m = np.min(x)
        if self.__m<1:
            x += np.abs(self.__m)
        self.__x = np.log1p(x)

        if self.__max_p is not None and self.__max_p>1:
            self.__stl = STL(self.__x, period=self.__max_p).fit()
            self.__x = self.__stl.trend + self.__stl.resid
            self.__snaivef = SeasonalNaive(self.__max_p)
            self.__snaivef.fit(self.__stl.seasonal)
            
        self.__queue = Queue(self.__x[-self.__window_length:])

        (X, y) = preprocess(self.__x, self.__window_length)
        self.__model.fit(X, y)

    def predict(self, h:int)->np.ndarray:
        predictions = []
        for i in range(h):
            q = self.__queue.as_array()
            x0 = q[-1]
            q -= x0
            f = self.__model.predict(q.reshape(-1, 1).T)[0]+x0
            predictions.append(f)
            self.__queue.dequeue()
            self.__queue.enqueue(f)

        predictions = np.array(predictions)

        if self.__snaivef is not None:
            predictions += self.__snaivef.predict(h=h)
            
        predictions = np.expm1(predictions)
        if self.__m<1:
            predictions -= np.abs(self.__m)
            
        return predictions

