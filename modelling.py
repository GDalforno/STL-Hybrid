import warnings
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from itertools import product
from forecasting import TimeSeriesForecaster

import rpy2.robjects.numpy2ri
from rpy2 import robjects
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
importr("forecast")
warnings.simplefilter("ignore")


def fit_r(command:str, ts:tuple)->np.ndarray:
    x = ts[0]
    h = len(ts[1])
    max_p = ts[2]
    
    robjects.r.assign("x", x)
    robjects.r.assign("x", robjects.r(f"ts(x, frequency={max_p})"))
    robjects.r.assign("fit", robjects.r(command))
    return np.array(robjects.r(f"forecast(fit, h={h})$mean"))


def get_window_length(max_p:int)->np.ndarray:
    max_p = max_p if max_p>3 else 10
    max_p = max_p if max_p%2==0 else max_p+2
    return np.arange(3, max_p, 2)


def fit_python(forecaster:str, ts:tuple)->np.ndarray:
    x = ts[0]
    h = len(ts[1])
    
    s, q = temporal_train_test_split(x, test_size=0.05)
    error = np.float("inf")
    len_q = len(q)
    f_star = None
    H = {
        "linear_svr":linear_svr_hp_space(ts, False),
        "stl_linear_svr":linear_svr_hp_space(ts, True),
        "bayesian_ridge":bayesian_hp_space(ts, False),
        "stl_bayesian_ridge":bayesian_hp_space(ts, True)
    }

    for f in H[forecaster]:
        #try:
            f.fit(s)
            z = f.predict(len_q)
            current_error = mean_absolute_error(q, z)
            if current_error < error:
                error = current_error
                f_star = f 
#         except:
#             continue
    if f_star is None:
        return np.repeat(np.nan, h)
    else:
        try:
            f_star.fit(x)
            return f_star.predict(h)
        except:
            return np.repeat(np.nan, h)


def linear_svr_hp_space(ts:tuple, decompose:bool)->iter:
    max_p = ts[2]
    
    window_length = get_window_length(max_p)
    iterator = product(np.arange(0.25, 1.25, 0.25), window_length)

    for (C, window_length) in iterator:
        params = {
            "model":LinearSVR(
                C=C, 
                dual=False, 
                loss="squared_epsilon_insensitive"),
            "window_length":window_length
        }
        if decompose:
            params["max_p"] = max_p

        yield TimeSeriesForecaster(**params)


def bayesian_hp_space(ts:tuple, decompose:bool)->iter:
    max_p = ts[2]
    
    for window_length in get_window_length(max_p):
        params = {
            "model":BayesianRidge(),
            "window_length":window_length
        }
        if decompose:
            params["max_p"] = max_p

        yield TimeSeriesForecaster(**params)


def autoarima(ts:tuple)->np.ndarray:
    return fit_r("auto.arima(x)", ts)


def autoets(ts:tuple)->np.ndarray:
    return fit_r("ets(x)", ts)


def linear_svr(ts:tuple)->np.ndarray:
    return fit_python("linear_svr", ts)


def stl_linear_svr(ts:tuple)->np.ndarray:
    return fit_python("stl_linear_svr", ts)


def bayesian_ridge(ts:tuple)->np.ndarray:
    return fit_python("bayesian_ridge", ts)


def stl_bayesian_ridge(ts:tuple)->np.ndarray:
    return fit_python("stl_bayesian_ridge", ts)

