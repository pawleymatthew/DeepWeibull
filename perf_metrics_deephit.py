import numpy as np
import pandas as pd

from scipy.integrate import simps

from pycox.evaluation import EvalSurv

def deep_hit_metrics_preprocess(test_result):

    """
    DESCRIPTION:
        The output of deep_hit is a test_result dataframe.
        The columns of this dataframe are ["t_0","t_1",...,"t_max","time","status"].
        I need to convert this back to the format this can be used for pycox.evaluation module.
    INPUT
        - test_result : dataframe with format of output of deep_hit()
    OUTPUT
        - surv : survival dataframe consistent with pycox.evaluation
        - test_time : the event/censoring times of the test set
        - test_status : the event indicator of the test set
    """
    surv = test_result.copy()
    test_time = np.array(surv.pop("time"))
    test_status = np.array(surv.pop("status"))
    surv = surv.transpose()
    surv = surv.reset_index(drop=True)
    
    deep_hit_eval_surv = EvalSurv(surv, test_time, test_status, censor_surv='km') 

    return deep_hit_eval_surv

def c_td_deep_hit(test_result):

    """
    DESCRIPTION: 
        Computes the time-dependent concordance index given a set of survival data and corresponding DeepHit predicted survival curves.
    INPUT:
        - test_result : dataframe with format of output of deep_hit()
    OUTPUT:
        - the time-dependent concordance index
    """
    
    deep_hit_eval_surv = deep_hit_metrics_preprocess(test_result)

    return deep_hit_eval_surv.concordance_td('antolini')


def int_brier_deep_hit(test_result, int_points=100):

    """
    DESCRIPTION: 
        Computes the integrated Brier score given a set of survival data and corresponding DeepHit predicted survival curves.
        The integration is performed numerically using Simpson's rule.
        The number of points in the partition should be sufficiently large for a good estimate.
        The timepoints are taken between the min and max of the "time" column.
        The (t,BS(t)) pairs are also an output, so a plot.
    INPUT:
        - test_result : dataframe with format of output of deep_hit()
        - int_points : the number of timepoints t for the numerical integration.
    OUTPUT:
        - int_brier : the integrated Brier score
        - t_vals : the timepoints using for the computation
        - brier_t_vals : the BS(t) scores for each t in t_vals
    """

    # create partition of timepoints to compute BS(t) over
    T_min = test_result["time"].min() # min. time
    T_max = test_result["time"].max() # max. time
    t_vals = np.linspace(T_min,T_max,num=int_points) # partition [T_min,T_max] with int_points sub-intervals

    # create EvalSurv object
    deep_hit_eval_surv = deep_hit_metrics_preprocess(test_result)

    # compute BS(t) as each point
    b_t = deep_hit_eval_surv.brier_score(t_vals)
    # integrate (by Simpson's rule) and divide by total width of time interval (i.e. T_max - T_min)
    int_b = float(simps(b_t, t_vals)/(T_max - T_min))

    return ({
            "int_brier" : int_b,
            "t_vals" : t_vals,
            "brier_t_vals" : b_t})

