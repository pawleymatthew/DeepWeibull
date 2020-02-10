import math
import numpy as np
import pandas as pd

import scipy.optimize

from pycox.evaluation import EvalSurv

from weibull_dist import weibull_surv


def regression_weibull_loglkhd(theta_a, theta_b, df):

    def div(x,y, eps=1e-10):
        return np.divide(x,y+eps)

    def log(x, eps=1e-10):
        return np.log(x+eps)
    
    x = df.drop(["time", "status"], axis=1) # Pandas dataframe of covariate columns
    a = theta_a[0] + x.dot(np.array(theta_a[1:])) # alpha = theta_0 + < theta[1:p], x >
    b = theta_b[0] + x.dot(np.array(theta_b[1:]))
    
    if True in (item <= 0 for item in a): # all alpha values must be positive for l to be defined
        return -1 * math.inf # L=0 so l=-infinity
   
    time_norm = div(df["time"], a)
    l = np.sum(df["status"] * log(div(b,a) * np.power(time_norm, b-1)))
    l -= np.sum(np.power(time_norm, b)) # second sum of l
    return l

def regression_weibull(dataset):

    """
    Paths to input and output files
    """
    train_path = "datasets/" + dataset + "_data/" + dataset + "_train_df.csv" # training set data
    test_path = "datasets/" + dataset + "_data/" + dataset + "_test_df.csv" # test set data
 
    """
    Read in the appropriate training and test sets.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # intialise parameters for the model
    p = train_df.shape[1] - 2 # number of covariates (number of columns excluding time, status)
    mean_time = train_df["time"].mean() # compute mean event time for estimate of alpha
    init_theta_a = [mean_time] + [0]*p # [*,0,...,0]
    init_theta_b = [1] + [0]*p # [1,0,...,0]
    init_params = init_theta_a + init_theta_b

    # fit the model
    fun = lambda x: -1 * regression_weibull_loglkhd(x[:len(x)//2], x[len(x)//2:], train_df) # wrapper for negative log-lkhd function
    res = scipy.optimize.minimize(fun, x0=init_params) # minimise negative log-lkhd

    # make predictions on the test set
    parameters = res.x
    half_length = len(parameters)//2
    theta_a = parameters[:half_length]
    theta_b = parameters[half_length:]
    test_result = test_df.copy()
    x = test_result.drop(["time", "status"], axis=1)
    test_result["pred_alpha"] = theta_a[0] + x.dot(np.array(theta_a[1:])) # alpha = theta_0 + < theta[1:p], x >
    test_result["pred_beta"] = theta_b[0] + x.dot(np.array(theta_b[1:])) # beta = phi_0 + < phi[1:p], x >
    # if any of the predicted alpha/beta are negative, set equal to small eps
    eps = 1e-8
    test_result["pred_alpha"] = np.maximum(test_result["pred_alpha"], eps*np.ones(len(test_result["pred_alpha"])))
    test_result["pred_beta"] = np.maximum(test_result["pred_beta"], eps*np.ones(len(test_result["pred_beta"])))

    """
    Create EvalSurv object
    """
    t_max = train_df["time"].max()
    num_vals = max(math.ceil(t_max), 5000)
    t_vals = np.linspace(0, t_max, num_vals)
    surv = weibull_surv(t_vals, test_result["pred_alpha"], test_result["pred_beta"])
    surv = pd.DataFrame(data=surv, index=t_vals)

    test_time = test_df['time'].values
    test_status = test_df['status'].values

    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')
    
    return ({"parameters" : parameters, "test_result" : test_result, "ev" : ev }) 
