import math
import numpy as np
import pandas as pd
import scipy.optimize

"""
Inputs:
    - df : a Pandas dataframe (N rows) with p feature columns and outcome columns called 'time' and 'status'
    - theta: list of the p+1 parameters [theta_0,...,theta_p]
    - phi: the phi parameter (must be >0)
Outputs:
    - the log-likelihood (note: NOT negative log-likelihood)
"""
def simple_model_one_loglkhd(theta_a, theta_b, df):
    
    eps = 1e-8

    x = df.drop(["time", "status"], axis=1) # Pandas dataframe of covariate columns
    a = theta_a[0] + x.dot(np.array(theta_a[1:])) # alpha = theta_0 + < theta[1:p], x >
    b = theta_b
    
    if True in (item <= 0 for item in a): # all alpha values must be positive for l to be defined
        return -1 * math.inf # L=0 so l=-infinity
   
    time_norm = np.divide(df["time"], a+eps)
    l = np.sum(df["status"] * np.divide(b,a+eps) * np.power(time_norm, b-1))
    l -= np.sum(np.power(time_norm, b)) # second sum of l
    return l

"""
Inputs:
    - train_df: a Pandas dataframe as in output of make_train_test()
    - test_df: a Pandas dataframe as in output of make_train_test()
    - init_params: parameter values to initialise the optimisation 
Outputs:
    - parameters: the parameters [theta_0,...,theta_p,phi] that minimise the negative log-likelihood
    - test_result: the predictions of the fitted model on the test set
"""

def simple_model_one(train_df, test_df):
    
    # intialise parameters for the model
    p = train_df.shape[1] - 2 # number of covariates = number of columns excluding time, status
    mean_time = train_df["time"].mean() # compute mean event time for estimate of alpha
    init_theta_a = [mean_time] + [0]*p # initial guess of alpha parameters is [*,0,...,0]
    init_theta_b = [1] # initial guess of beta parameters is [1]
    
    # fit the model
    init_params = init_theta_a + init_theta_b # init_params = [*,0,...,0,1]
    fun = lambda x: -1 * simple_model_one_loglkhd(x[:-1], x[-1], train_df) # wrapper function: maps x to -logL(x|train_df)
    res = scipy.optimize.minimize(fun, x0 = init_params) # minimise negative log-lkhd
    
    # make predictions on the test set
    parameters = res.x 
    theta_a = parameters[0:-1] # all but last parameter
    theta_b = parameters[-1] # last parameter
    test_result = test_df.copy()
    x = test_result.drop(["time", "status"], axis=1)
    test_result["pred_alpha"] = theta_a[0] + x.dot(np.array(theta_a[1:])) # alpha = thetaa_0 + < thetaa[1:p], x >
    test_result["pred_beta"] = theta_b # beta = theta_b
    
    return ({
            "parameters" : theta_a + theta_b, # the parameter values that minimise the negative log-lkhd
            "test_result" : test_result # the predictions of the fitted model on the test set
            }) 

"""
Inputs:
    - df : a Pandas dataframe (N rows) with p feature columns and outcome columns called 'time' and 'status'
    - theta: list of the p+1 parameters [theta_0,...,theta_p]
    - phi: list of the p+1 parameters [phi_0,...,phi_p]
Outputs:
    - the log-likelihood (note: NOT negative log-likelihood)
"""
def simple_model_two_loglkhd(theta_a, theta_b, df):
    
    x = df.drop(["time", "status"], axis=1) # Pandas dataframe of covariate columns
    a = theta_a[0] + x.dot(np.array(theta_a[1:])) # alpha = thetaa_0 + < thetaa[1:p], x >
    b = theta_b[0] + x.dot(np.array(theta_b[1:])) # beta = thetab_0 + < thetab[1:p], x >
    
    if True in (item <= 0 for item in pd.concat([a,b])): # all alpha and beta values must be positive
        return - math.inf # L=0 so l=-infinity  
    
    l = np.sum(df["status"] * np.log(b * np.power(a,-b) * np.power(df["time"],b-1))) # first sum of l
    l -= np.sum(np.power(np.divide(df["time"],a), b)) # second sum of l
    return l

"""
Inputs:
    - train_df: a Pandas dataframe as in output of make_train_test()
    - test_df: a Pandas dataframe as in output of make_train_test() 
Outputs:
    - parameters: the parameters [theta_0,...,theta_p,phi] that minimise the negative log-likelihood
    - test_result: the predictions of the fitted model on the test set
"""

def simple_model_two(train_df, test_df):
    
    # intialise parameters for the model
    p = train_df.shape[1] - 2 # number of covariates (number of columns excluding time, status)
    mean_time = train_df["time"].mean() # compute mean event time for estimate of alpha
    init_theta_a = [mean_time] + [0]*p # [*,0,...,0]
    init_theta_b = [1] + [0]*p # [1,0,...,0]
    init_params = init_theta_a + init_theta_b

    # fit the model
    fun = lambda x: -1 * simple_model_two_loglkhd(x[:len(x)//2], x[len(x)//2:], train_df) # wrapper for negative log-lkhd function
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
    return ({
            "parameters" : parameters, # the parameter values that minimise the negative log-lkhd
            "test_result" : test_result # the predictions of the fitted model on the test set
            }) 
