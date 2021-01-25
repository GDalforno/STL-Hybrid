import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf


def plot(data:list, title:str)->None:
    training, testing = data[0], data[1]
    n_train, n_test = len(training), len(testing)
    plt.figure(figsize=(13, 5))
    plt.title(title)
    plt.plot(training, color="blue", label="training data")
    plt.plot(np.arange(n_train, n_train+n_test), testing, color="cornflowerblue", label="testing data")
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(0, n_train+n_test, 12))
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
    axs[1].plot(res.seasonal, color="royalblue")
    axs[1].grid()
    axs[2].set_title("Residual")
    axs[2].plot(res.resid, color="darkblue")
    axs[2].grid()
    plt.xticks(np.arange(0, len(data), 12))
    plt.show()

    return res


def acf(data:np.ndarray)->None:
    plt.rc("figure", figsize=(10, 5)) 
    _=plot_acf(data)
    plt.grid()

    
def plot_dist(data:np.ndarray)->None:
    plt.title("Residuals")
    sns.histplot(data, bins=25, kde=True)
    plt.grid()
    plt.show()
