import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from weibull_dist import weibull_survival

"""
Inputs:
    - i : index of an individual
    - time : the event times
    - status : the censoring statuses
Outputs:
    - comparable_index : list of individuals {j} such that (i,j) are comparable pairs
"""

def comparable_index(i, time, status):

    if status[i] == 0:
        return [] # no comparable pairs (i,j)
    else: 
        return time[time > time[i]].index.tolist() # all j such that t_i < t_j

"""
Inputs:
    - i : the index of an individual 
    - j : the index of a different individual (such that (i,j) is comparable)
    - time_i, time_j : the event/censoring times
    - alpha_i, beta_i : the Weibull parameters of i (under the model, i.e. pred_alpha, pred_beta)
    - alpha_j, beta_j : the Weibull parameters of j (under the model, i.e. pred_alpha, pred_beta)
Outputs:
    - concordant : Boolean value, concordant (True) or not (False)
"""

def concordant(i, j, time_i, alpha_i, beta_i, time_j, alpha_j, beta_j):

    sf_i = weibull_survival(time_i, alpha_i, beta_i) # S(t_i ; a_i, b_i)
    sf_j = weibull_survival(time_i, alpha_j, beta_j) # S(t_i ; a_j, b_j)
    
    return sf_i < sf_j

"""
Inputs:
    - test_result : a Pandas dataframe of the test result
    - sample_frac : for estimating the c-index based on sample of individuals (if speed is an issue)
Outputs:
    - the time-dependent concordance index
"""

def time_dependent_concordance(test_result, sample_frac=1):

    # take a sample of the data (and reset index)
    test_result = test_result.sample(frac=sample_frac).reset_index()

    # make lists from the columns
    time = test_result["time"]
    status = test_result["status"]
    alpha = test_result["pred_alpha"]
    beta = test_result["pred_beta"]

    # list of all comparable pairs (each pair as list)
    pairs = [[i,j] for i in range(test_result.shape[0]) for j in comparable_index(i,time,status)]

    # number of concordant pairs divided by number of comparable pairs
    c = sum([concordant(pair[0],pair[1],time[pair[0]],alpha[pair[0]],beta[pair[0]],time[pair[1]],alpha[pair[1]],beta[pair[1]]) for pair in pairs])
    c /= len(pairs)

    return c
