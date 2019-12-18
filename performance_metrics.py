import numpy as np
import pandas as pd
from weibull_dist import weibull_survival

"""
Input:
    - test_result : Pandas dataframe of time, status, pred_alpha, pred_beta
Output:
    - the time-dependent concordance index
"""

def time_dependent_concordance(test_result):

    test_result.sort_values(by="time", ascending=True, inplace=True) # sort by time column
    N = test_result.shape[0]

    # initialise counts of comparable and concordant pairs
    comp = 0
    conc = 0

    for i in range(N-1):
        if test_result.at[test_result.index[i], "status"] == 1:

            t_i = test_result.at[test_result.index[i], "time"]
            alpha_i = test_result.at[test_result.index[i], "pred_alpha"]
            beta_i = test_result.at[test_result.index[i], "pred_beta"]

            p = weibull_survival(t_i, alpha_i, beta_i) # save S(t_i|x_i) in memory

            for j in range(i+1,N): # for individuals with surival time >= t_i
                
                t_j = test_result.at[test_result.index[j], "time"]
                
                if t_i < t_j: # this step is redundant if there are no ties

                    comp += 1 # (i,j) are comparable

                    alpha_j = test_result.at[test_result.index[j], "pred_alpha"]
                    beta_j = test_result.at[test_result.index[j], "pred_beta"]

                    if p < weibull_survival(t_i, alpha_j, beta_j):

                        conc += 1 # (i,j) are concordant

    return conc/comp

"""
Input:
    - test_result : Pandas dataframe of time, status, pred_alpha, pred_beta
    - t : the fixed time t to evaluate Brier score with respect to 
Output:
    - the Brier score (t fixed, not integrated Brier score)
"""

def brier(test_result, t):

    brier_df = test_result.copy()

    brier_df["dead_by_t"] = (brier_df["time"] <= t) and (brier_df["status"] == 1) # death occurred by time t
    brier_df["survive_t"] = brier_df["time"] > t
    
    brier_score = 0
    N = brier_df.shape[0]

    for i in range(N):

        t_i = test_result.at[test_result.index[i], "time"]
        alpha_i = test_result.at[test_result.index[i], "pred_alpha"]
        beta_i = test_result.at[test_result.index[i], "pred_beta"]
        S_i = weibull_survival(t_i, alpha_i, beta_i)

        brier_score += S_i**2 * test_result.at[test_result.index[i], "dead_by_t"] 
        brier_score += (1-S_i)**2 * test_result.at[test_result.index[i], "survive_t"] 

    brier_score /= N

    return brier_score

