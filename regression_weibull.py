import math
import numpy as np
import pandas as pd
import scipy.optimize
from pycox.evaluation import EvalSurv
from weibull_dist import weibull_surv

def regression_gumbel_loglkhd(theta, sigma, df):
    
    x = df.drop(["time", "status"], axis=1) # Pandas dataframe of covariate columns
    mu = theta[0] + x.dot(np.array(theta[1:])) # mu = theta_0 + < theta[1:p], x >
     
    l = np.sum(df["status"] * (np.divide(np.subtract(np.log(df["time"]),mu),sigma))-np.log(sigma)) # first sum of l
    l -= np.sum(np.exp(np.divide(np.subtract(np.log(df["time"]),mu),sigma))) # second sum of l
    
    return l


def regression_weibull(dataset, split):

    """
    Paths to input and output files
    """
    train_path = "datasets/" + dataset + "/train_" + str(split) + ".csv" # training set data
    test_path = "datasets/" + dataset + "/test_" + str(split) + ".csv" # test set data

    """
    Read in the appropriate training and test sets (dataset name and split index)
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # intialise parameters for the model
    p = train_df.shape[1] - 2 # number of covariates (number of columns excluding time, status)
    median_log_time = np.log(train_df["time"]).median() # compute median time for estimate of alpha
    init_theta = [median_log_time] + [0]*p # [*,0,...,0] # for mu=median log time
    init_sigma = [1]
    init_params = init_theta + init_sigma

    # fit the model
    fun = lambda x: -1 * regression_gumbel_loglkhd(x[:-1], x[-1], train_df) # wrapper for negative log-lkhd function
    res = scipy.optimize.minimize(fun, x0=init_params) # minimise negative log-lkhd

    # make predictions on the test set
    parameters = res.x
    theta = parameters[:-1]
    sigma = parameters[-1]
    test_result = test_df.copy()
    x = test_result.drop(["time", "status"], axis=1)
    test_result["pred_alpha"] = np.exp(theta[0] + x.dot(np.array(theta[1:]))) # alpha = exp(mu)
    test_result["pred_beta"] = 1/sigma # beta = 1/sigma

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

