import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf


def plot(data:np.ndarray, title:str)->None:
    plt.figure(figsize=(13, 5))
    plt.title(title)
    plt.plot(data, color="blue")
    plt.grid()
    plt.xticks(np.arange(0, len(data), 12))
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.show()


def plot_stl(data:np.ndarray)->DecomposeResult:
    stl = STL(data, period=12)
    res = stl.fit()
    fig, axs = plt.subplots(3, sharex=True, figsize=(10, 10))
    axs[0].set_title("Trend")
    axs[0].plot(res.trend, color="blue")
    axs[0].grid()
    axs[1].set_title("Seasonal")
    axs[1].plot(res.seasonal, color="red")
    axs[1].grid()
    axs[2].set_title("Residual")
    axs[2].plot(res.resid, color="black")
    axs[2].grid()
    plt.xticks(np.arange(0, len(data), 12))
    plt.show()

    return res


def acf(data:np.ndarray)->None:
    plt.rc("figure", figsize=(10, 5)) 
    _=plot_acf(data)
    plt.grid()


def plot_predictions(y_test:np.ndarray, y_pred:np.ndarray, model:str)->None:
    plt.plot(y_test, label="Testing", color="blue")
    plt.plot(y_pred, label=model, color="red")
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

