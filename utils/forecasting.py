import numpy as np
from sklearn.base import RegressorMixin
from statsmodels.tsa.seasonal import STL
from pmdarima.arima import ndiffs
from utils import*


class TimeSeriesForecaster:
    def __init__(
            self, model:RegressorMixin, window_length:int, detrend:bool,
            period:int=None):
        self.__window_length = window_length  
        self.__detrend = detrend
        self.__period = period 
        self.__model = model
        self.__snaivef = None 
        self.__stl = None 
        self.__x = None
        self.__m = None
        self.__queue = None
        self.__detrend = None

    def fit(self, x:np.ndarray)->None:
        self.__m = np.min(x)
        if self.__m<1:
            x += np.abs(self.__m)
        self.__x = np.log1p(x)

        if self.__period is not None:
            self.__stl = STL(self.__x, period=self.__period).fit()
            self.__x = self.__stl.trend + self.__stl.resid
            self.__snaivef = SeasonalNaive(self.__period)
            self.__snaivef.fit(self.__stl.seasonal)

        if self.__detrend:
            q = self.__x[-self.__window_length-1:]
        else:
            q = self.__x[-self.__window_length:]
        self.__queue = Queue(q)

        
        (X, y) = preprocess(self.__x, self.__window_length, self.__detrend)
        self.__model.fit(X, y)

    def predict(self, h:int)->np.ndarray:
        predictions = []
        for i in range(h):
            q = self.__queue.as_array()
            if self.__detrend:
                x0 = q[-1]
                q = q[:-1] - x0
                f = self.__model.predict(q.reshape(-1, 1).T)[0]+x0
            else:
                f = self.__model.predict(q.reshape(-1, 1).T)[0]
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


class SeasonalNaive:
    def __init__(self, period:int):
        self.__period = period
        self.__data = None
    
    def fit(self, data:np.ndarray)->None:
        self.__data = data[-self.__period:]
        
    def predict(self, h:np.int)->np.ndarray:
        y = np.tile(self.__data, h//self.__period+1)
        return y[:h]


class Naive:
    def __init__(self):
        self.__x = None
    
    def fit(self, data:np.ndarray)->None:
        self.__x = data[-1]
        
    def predict(self, h:np.int)->np.ndarray:
        return np.repeat(self.__x, h)

