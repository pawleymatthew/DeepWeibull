import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf

"""
Inputs:
    N - number of individuals, a positive integer. 
    N_c - number of censored individuals, an integer in 0 <= N_c < N.
 Outputs:
    A pandas dataframe with N rows and 5 columns called "x1","x2","x3","time","status".
 Description:
    Creates a dataset of N individuals, N_c of whom are censored. Each individual has three (iid) covariates. 
    The outcomes, i.e. time to event and censoring indicator, are simulated from a Weibull distribution.
    The parameters of the Weibull distribution are independent of the covariates.
    Can be used to test whether a model is able to learn the absence of a relationship between features and target. 
"""

def synthdata_noise(N,N_c):
    x = np.random.normal(loc=0.0, scale=1.0, size=(N,3)) # sim covariates (3 per individual)
    a = 10e-2 + abs(np.random.normal(loc=50.0, scale=10.0, size=N)) # sim alphas (and ensure >0)
    b = 10e-2 + abs(np.random.normal(loc=1.0, scale=0.05, size=N)) # sim betas (and ensure >0)
    time = a * np.random.weibull(b) # sim event times from Weibull(a,b)
    status = np.ones(N) # initialise status indicator (all dead)
    censored = random.sample(range(0,N), N_c) # randomly select N_c individuals to be censored
    status[censored] = 0 # set status indicator for censored patients to 0
    time[censored] = np.random.uniform(low=0,high=time[censored]) # simulate censoring time Uniform(0,old event time)
    data = {
        'x1':[item[0] for item in x], # covariate 1
        'x2':[item[1] for item in x], # covariate 2
        'x3':[item[2] for item in x], # covariate 3
        'time': time, # simulated event/censoring time
        'status': status, # 0=censored, 1=dead
        'true_alpha' : a, # true alpha value used to simulate death time
        'true_beta' : b # true beta value used to simulate death time
    }
    df = pd.DataFrame(data)
    return df

"""
Inputs:
    N - number of individuals, a positive integer. 
    N_c - number of censored individuals, an integer in 0 <= N_c < N.
 Outputs:
    A pandas dataframe with N rows and 5 columns called "x1","x2","x3","time","status".
 Description:
    Creates a dataset of N individuals, N_c of whom are censored. Each individual has three (iid) covariates. 
    The outcomes, i.e. time to event and censoring indicator, are simulated from Weibull distribution.
    The parameters of the Weibull distribution are given by a function of the covariates.
    Can be used to test whether a model is able to learn how a model learns a mapping from covariates to outcomes.
"""

def synthdata_custom(N,N_c):
    x = np.random.normal(loc=0.0, scale=1.0, size=(N,3)) # sim covariates (3 per individual)
    a = [max(10e-2,50+(10/math.sqrt(3.0))*(item[0]**2+item[1]-1)) for item in x] # compute alphas (ensure >0)
    b = [max(10e-2,1+0.05*item[2]) for item in x] # compute betas (ensure >0)
    time = a * np.random.weibull(b) # sim event times from Weibull(a,b)
    status = np.ones(N) # initialise status indicator (all dead)
    censored = random.sample(range(0,N), N_c) # randomly select N_c individuals to be censored
    status[censored] = 0 # set status indicator for censored to 0
    time[censored] = np.random.uniform(low=0,high=time[censored]) # simulate censoring time Uniform(0,old event time)
    data = {
        'x1':[item[0] for item in x], # covariate 1
        'x2':[item[1] for item in x], # covariate 2
        'x3':[item[2] for item in x], # covariate 3
        'time': time, # simulated event/censoring time
        'status': status, # 0=censored, 1=dead
        'true_alpha' : a, # true alpha value used to simulate death time
        'true_beta' : b # true beta value used to simulate death time
    }
    df = pd.DataFrame(data)
    return df

"""
Inputs:
    - df: a Pandas dataframe (N rows) with feature columns (names don't matter) and outcome columns called 'time' and 'status'
    - train_frac: the fraction of rows to be allocated to the training set
Outputs:
    - a named list of Pandas dataframes:
        - 'train_df' containing the training set features and outcomes (columns: x1, x2, x3, time, status)
        - 'test_df' containing the test set features features and outcomes (columns: x1, x2, x3, time, status)
Description:
    Creates the training and test sets in the form (tensors) required for use in the model(s).
"""
def make_train_test(df, train_frac):
    # randomly select inds to go in train/test sets
    train_df = df.sample(frac=train_frac)
    test_df = df.drop(train_df.index)
    # return named list of dataframes
    return ({"train_df" : train_df,
             "test_df" : test_df})
