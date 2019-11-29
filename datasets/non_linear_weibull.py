import numpy as np
import pandas as pd
import random

from make_train_test import make_train_test

"""
Description:
    - Simulate a dataset where the event times are (potentially censored) realisations of a Weibull distribution whose parameters are a non-linear function of the covariates.
Inputs:
    - N : a positive integer, the number of individuals
    - N_c : an integer in [0,N], number of censored individuals
Outputs:
    - df :  a pd dataframe with N rows and columns ("x1",..."x_","time","status","true_alpha","true_beta)
"""

def non_linear_weibull_data(N, N_c):
    
    x = np.random.normal(loc=0.0, scale=1.0, size=(N,3)) # sim covariates (3 per individual)
    df = pd.DataFrame(x, columns=['x{}'.format(i) for i in range(1, 4)]) # make dataframe

    # determine the Weibull parameters
    alpha = 50 + 2 * np.sign(x[:,0]) * np.square(x[:,0]) + 2 * np.multiply(x[:,0],x[:,1])
    beta = 1.1 + 0.03 * np.sign(x[:,0]) * np.square(x[:,0]) + 0.02 * np.multiply(x[:,0],x[:,2])

    if True in (item <= 0 for item in np.concatenate([alpha,beta])): # all alpha and beta values must be positive
        raise ValueError('Weibull parameters must be positive.') 

    # simulate event times and censoring information/times
    time = alpha * np.random.weibull(beta, size=N)
    status = np.ones(N)
    censored = random.sample(range(0,N), N_c)
    status[censored] = 0
    time[censored] = np.random.uniform(low=0,high=time[censored])

    # add to dataframe
    df["time"] = time 
    df["status"] = status
    df["true_alpha"] = alpha
    df["true_beta"] = beta

    return df

"""
Simulate the data and write to a csv file.
"""
# set inputs
N = 30000
N_c = 15000

# create dataframe
df = non_linear_weibull_data(N, N_c)

# save as csv file
df.to_csv(r"datasets/non_linear_weibull_df.csv", index=False)

"""
Split into training and test sets and write these to csv files.
"""

train_frac = 0.8

# make the train/test sets
sets = make_train_test(df, train_frac)

# write to csv files
sets["train_df"].to_csv(r"datasets/non_linear_weibull_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/non_linear_weibull_test_df.csv", index=False)
