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
def simple_model_one_loglkhd(theta, phi, df):
    
    x = df.drop(["time", "status", "true_alpha", "true_beta"], axis=1) # Pandas dataframe of covariate columns
    a = theta[0] + x.dot(np.array(theta[1:])) # alpha = theta_0 + < theta[1:p], x >
    
    if True in (item <= 0 for item in a): # all alpha values must be positive for l to be defined
        return - math.inf # L=0 so l=-infinity
    
    l = np.sum(df["status"] * np.log(phi * np.power(a,-phi) * np.power(df["time"],phi-1))) # first sum of l
    l -= np.sum(np.power(df["time"]/a, phi-1)) # second sum of l

    return l

"""
Inputs:
    - train_df: a Pandas dataframe as in output of make_train_test()
    - test_df: a Pandas dataframe as in output of make_train_test()
    - init_params: parameter values to initialise the optimisation 
Outputs:
    - parameters: the parameters [theta_0,...,theta_p,phi] that minimise the negative log-likelihood
    - convergence_success: whether the optimiser converged successfully
    - objective_function: the value of the objective function (negative log-likelihood) at the MLE
    - test_result: the predictions of the fitted model on the test set
"""

def simple_model_one(train_df, test_df, init_theta, init_phi):
    # fit the model
    fun = lambda x: -1 * simple_model_one_loglkhd(x[:-1], x[-1], train_df) # wrapper function: maps x to -logL(x|train_df)
    res = scipy.optimize.minimize(fun, x0=init_theta + init_phi) # minimise negative log-lkhd
    # make predictions on the test set
    parameters = res.x # the optimal parameter values
    theta = parameters[0:-1] 
    phi = parameters[-1]
    test_result = test_df.copy()
    x = test_result.drop(["time", "status", "true_alpha", "true_beta"], axis=1)
    test_result["pred_alpha"] = theta[0] + x.dot(np.array(theta[1:])) # alpha = theta_0 + < theta[1:p], x >
    test_result["pred_beta"] = phi # beta = phi
    return ({
            "parameters" : parameters, # the parameter values that minimise the negative log-lkhd
            "convergence_success" : res.success, # Boolean: whether the optimiser converged successfully
            "objective_function" : res.fun, # optimal value of the objective function
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
def simple_model_two_loglkhd(theta, phi, df):
    
    x = df.drop(["time", "status", "true_alpha", "true_beta"], axis=1) # Pandas dataframe of covariate columns
    a = theta[0] + x.dot(np.array(theta[1:])) # alpha = theta_0 + < theta[1:p], x >
    b = phi[0] + x.dot(np.array(phi[1:])) # beta = phi_0 + < phi[1:p], x >
    
    if True in (item <= 0 for item in pd.concat([a,b])): # all alpha and beta values must be positive
        return - math.inf # L=0 so l=-infinity
    
    l = np.sum(df["status"] * np.log(b * np.power(a,-b) * np.power(df["time"],b-1))) # first sum of l
    l -= np.sum(np.power(df["time"]/a, b-1)) # second sum of l
    return l

"""
Inputs:
    - train_df: a Pandas dataframe as in output of make_train_test()
    - test_df: a Pandas dataframe as in output of make_train_test()
    - init_theta: theta parameter values to initialise the optimisation
    - init_phi: phi parameter values to initialise the optimisation 
Outputs:
    - parameters: the parameters [theta_0,...,theta_p,phi] that minimise the negative log-likelihood
    - convergence_success: whether the optimiser converged successfully
    - objective_function: the value of the objective function (negative log-likelihood) at the MLE
    - test_result: the predictions of the fitted model on the test set
"""

def simple_model_two(train_df, test_df, init_theta, init_phi):
    # fit the model
    fun = lambda x: -1 * simple_model_two_loglkhd(x[:len(x)//2], x[len(x)//2:], train_df) # wrapper for negative log-lkhd function
    res = scipy.optimize.minimize(fun, x0=init_theta + init_phi) # minimise negative log-lkhd
    # make predictions on the test set
    parameters = res.x
    half_length = len(parameters)//2
    theta = parameters[:half_length]
    phi = parameters[half_length:]
    test_result = test_df.copy()
    x = test_result.drop(["time", "status", "true_alpha", "true_beta"], axis=1)
    test_result["pred_alpha"] = theta[0] + x.dot(np.array(theta[1:])) # alpha = theta_0 + < theta[1:p], x >
    test_result["pred_beta"] = phi[0] + x.dot(np.array(phi[1:])) # beta = phi_0 + < phi[1:p], x >
    return ({
            "parameters" : parameters, # the parameter values that minimise the negative log-lkhd
            "convergence_success" : res.success, # Boolean: whether the optimiser converged successfully
            "objective_function" : res.fun, # optimal value of the objective function
            "test_result" : test_result # the predictions of the fitted model on the test set
            }) 
