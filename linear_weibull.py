import numpy as np
import pandas as pd
import random
from make_train_test import make_train_test

"""
Description:
    - Simulate a dataset where the event times are (potentially censored) realisations of a Weibull distribution whose parameters are linear combinations of the covariates.
Inputs:
    - N : a positive integer, the number of individuals
    - N_c : an integer in [0,N], number of censored individuals
    - theta_a : a list, the regression parameters for the alpha Weibull parameter
    - theta_b : a list, the regression parameters for the bet Weibull parameter
Outputs:
     - df :  a pd dataframe with N rows and columns ("x1",..."x_","time","status")
"""

def linear_weibull_data(N, N_c, theta_a, theta_b):

    p = len(theta_a) - 1 # number of covariates (theta includes intercept term, so minus one)
    
    x = np.random.normal(loc=0.0, scale=1.0, size=(N,p)) # sim covariates (p per individual)
    df = pd.DataFrame(x, columns=['x{}'.format(i) for i in range(1, p+1)]) # make dataframe

    # determine the Weibull parameters
    alpha = theta_a[0] + x.dot(np.array(theta_a[1:]))
    beta = theta_b[0] + x.dot(np.array(theta_b[1:]))

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

    return df

"""
Simulate the data and write to a csv file.
"""

N = 30000 # number of individuals
N_c = 5000 # number of censored individuals
theta_a = [50, 8, -8, 0, 0] # alpha regression parameters
theta_b = [1.1, 0.1, 0, -0.1, 0] # beta regression parameters

# create dataframe
df = linear_weibull_data(N, N_c, theta_a, theta_b)

# save as csv file
df.to_csv(r"datasets/linear_weibull_data/linear_weibull_df.csv", index=False)

"""
Split into training and test sets and write these to csv files.
"""

train_frac = 0.9
sets = make_train_test(df, train_frac)

# write to csv files
sets["train_df"].to_csv(r"datasets/linear_weibull_data/linear_weibull_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/linear_weibull_data/linear_weibull_test_df.csv", index=False)

