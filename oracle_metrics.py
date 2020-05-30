import math
import numpy as np
import pandas as pd

from pycox.evaluation import EvalSurv
from weibull_dist import weibull_surv

from evaluation_metrics import c_index, brier_score


"""
Compute the oracle c-index and integrated Brier score.
Only for the Weibull synthetic datasets

dataset : dataset name
split : 1, 2 or 3
linear : is it linear Weibull or non-linear Weibull
"""

def oracle_metrics(dataset, split, linear=True):

    test_path = "datasets/" + dataset + "/test_" + str(split) + ".csv" # test set data
    test_df = pd.read_csv(test_path)

    # take only maximum of 10,000 individuals
    N = max(test_df.shape[0],10000)
    test_df = test_df.iloc[0:N]

    test_result = test_df.copy()
    x = test_result.drop(["time", "status"], axis=1)

    if linear==True :
        theta = [50, 25, -25, 0, 0, 0]
        test_result["pred_alpha"] = theta[0] + x.dot(np.array(theta[1:]))
        test_result["pred_beta"] = 1

    else:
        test_result["pred_alpha"] = x.apply(lambda row: 80 - 40*row.x0**2 + 30*row.x0*row.x1, axis=1)
        test_result["pred_beta"] = 1.1

    """
    Create EvalSurv object
    """
    t_max = test_df["time"].max()
    num_vals = max(math.ceil(t_max), 5000)
    t_vals = np.linspace(0, t_max, num_vals)
    surv = weibull_surv(t_vals, test_result["pred_alpha"], test_result["pred_beta"])
    surv = pd.DataFrame(data=surv, index=t_vals)

    test_time = test_df['time'].values
    test_status = test_df['status'].values

    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

    c = c_index(ev)
    ibs = brier_score(ev, dataset, split, int_points=100)["int_score"]

    return ({"oracle_c_index" : c, "oracle_int_brier" : ibs})


print(oracle_metrics("big_linear_weibull", 3, linear=True))