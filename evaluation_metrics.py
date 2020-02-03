import numpy as np
import pandas as pd

from pycox.evaluation import EvalSurv

def c_index(ev):

    return ev.concordance_td('antolini')


def brier_score(ev, dataset, int_points=100):

    """
    Get the test set.
    """
    test_path = "datasets/" + dataset + "_data/" + dataset + "_test_df.csv" # test set data
    test_df = pd.read_csv(test_path)

    """
    Make grid of time points.
    """

    time_min = test_df["time"].min() # shortest time in test set
    time_max = test_df["time"].max() # longest time in test set
    time_grid = np.linspace(time_min, time_max, int_points) # partition [t_min, t_max] into int_points

    """
    Compute scores and integrated score.
    """
    
    scores = ev.brier_score(time_grid) # get a grid of BS(t) values for t in time_grid
    int_score = ev.integrated_brier_score(time_grid) # get the integrated Brier score

    return ({"scores" : scores, "int_score" : int_score})

