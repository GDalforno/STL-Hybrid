import pandas as pd
from pmdarima.model_selection import train_test_split

FILES = (
    "bebida.data", "cbe_chocolate.data", "cbe_beer.data",
    "cbe_electricity_production.data", "chicken.data", "consumo.data",
    "darwin.data", "dow_jones.data", "energia.data", "global.data",
    "icv.data", "ipi.data", "latex.data", "lavras.data", "maine.data",
    "mprime.data", "osvisit.data", "ozonio.data", "pfi.data", 
    "reservoir.data", "stemp.data", "temperatura_c.data",
    "temperatura_u.data", "usa.data", "wine_fw.data", "wine_dw.data",
    "wine_sw.data", "wine_red.data", "wine_rose.data", 
    "wine_sparkling.data"
)
N_FILES = 30
TEST_SIZE = 0.05


def load_dataset(index:int)->tuple:
    assert index<N_FILES, "Index out of range"
    
    path = "../datasets/"+FILES[index]
    time_series = pd.read_csv(path, header=None).values.reshape(-1)
    y_train, y_test = train_test_split(time_series, test_size=TEST_SIZE)
    
    return (y_train, y_test)

