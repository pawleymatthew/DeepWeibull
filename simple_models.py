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
    
    x = df.drop(['time', 'status'], axis=1) # Pandas dataframe of covariate columns
    a = theta[0] + x.dot(np.array(theta[1:])) # alpha = theta_0 + < theta[1:p], x >
    
    if True in (item <= 0 for item in a): # all alpha values must be positive for l to be defined
        return - math.inf
    
    l = np.sum(df["status"] * np.log(phi * np.power(a,-phi) * np.power(df["time"],phi-1))) # first sum of l
    l -= np.sum(np.power(df["time"]/a, phi-1)) # second sum of l

    return l

"""
Inputs:
    - df: a Pandas dataframe (N rows) with feature columns (names don't matter) and outcome columns called 'time' and 'status'
Outputs:
    - mle: the parameters [theta_0,...,theta_p,phi] that minimise the negative log-likelihood
    - convergence_success: whether the optimiser converged successfully
    - objective_function: the value of the objective function (negative log-likelihood) at the MLE
"""

def simple_model_one(df, init_params):
    fun = lambda x: -1 * simple_model_one_loglkhd(x[:-1], x[-1], df) # wrapper for negative log-lkhd function
    res = scipy.optimize.minimize(fun, method='Nelder-Mead', x0=init_params) # minimise negative log-lkhd
    return ({
            "mle" : res.x, # the parameter values that minimise the negative log-lkhd
            "convergence_success" : res.success, # Boolean: whether the optimiser converged successfully
            "objective_function" : res.fun # optimal value of the objective function
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
    
    x = df.drop(['time', 'status'], axis=1) # Pandas dataframe of covariate columns
    a = theta[0] + x.dot(np.array(theta[1:])) # alpha = theta_0 + < theta[1:p], x >
    b = phi[0] + x.dot(np.array(phi[1:])) # beta = phi_0 + < phi[1:p], x >
    
    if True in (item <= 0 for item in pd.concat([a,b])): # all alpha and beta values must be positive
        return - math.inf
    
    l = np.sum(df["status"] * np.log(b * np.power(a,-b) * np.power(df["time"],b-1))) # first sum of l
    l -= np.sum(np.power(df["time"]/a, b-1)) # second sum of l
    return l

"""
Inputs:
    - df: a Pandas dataframe (N rows) with feature columns (names don't matter) and outcome columns called 'time' and 'status'
Outputs:
    - mle: the parameters [theta_0,...,theta_p,phi_0,...,phi_p] that minimise the negative log-likelihood
    - convergence_success: whether the optimiser converged successfully
    - objective_function: the value of the objective function (negative log-likelihood) at the MLE
"""

def simple_model_two(df, init_params):
    fun = lambda x: -1 * simple_model_two_loglkhd(x[:len(x)//2], x[len(x)//2:], df) # wrapper for negative log-lkhd function
    res = scipy.optimize.minimize(fun, method='Nelder-Mead', x0=init_params) # minimise negative log-lkhd
    return ({
            "mle" : res.x, # the parameter values that minimise the negative log-lkhd
            "convergence_success" : res.success, # Boolean: whether the optimiser converged successfully
            "objective_function" : res.fun}) # optimal value of the objective function