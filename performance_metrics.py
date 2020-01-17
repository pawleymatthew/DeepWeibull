import numpy as np
import pandas as pd
from weibull_dist import weibull_survival, weibull_int_hazard
from scipy.integrate import simps

"""
Input:
    - x,y : a pair of real numbers
    - sigma : a positive real number 
Output:
    - et
"""


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

            p = weibull_int_hazard(t_i, alpha_i, beta_i) # save M(t_i|x_i) in memory

            for j in range(i+1,N): # for individuals with surival time >= t_i
                
                t_j = test_result.at[test_result.index[j], "time"]
                
                if t_i < t_j: # this step is redundant if there are no ties

                    comp += 1 # (i,j) are comparable

                    alpha_j = test_result.at[test_result.index[j], "pred_alpha"]
                    beta_j = test_result.at[test_result.index[j], "pred_beta"]

                    if p > weibull_int_hazard(t_i, alpha_j, beta_j):

                        conc += 1 # (i,j) are concordant

    return float(conc/comp)

"""
Input:
    - test_result : Pandas dataframe of time, status, pred_alpha, pred_beta
    - t : the fixed time t to evaluate Brier score with respect to 
Output:
    - the Brier score (t fixed, not integrated Brier score)
"""

def brier(test_result, t):

    brier_df = test_result.copy()

    brier_df["dead_by_t"] = np.where((brier_df["time"]<=t) & (brier_df["status"]==1), 1, 0) # death occurred by time t
    brier_df["survive_t"] = np.where(brier_df["time"]>t, 1, 0) # survived to at least time t
    
    brier_score = 0
    N = brier_df.shape[0]

    for i in range(N):
        # P(T > t_i | alpha_i, beta_i)
        S_i = weibull_survival(brier_df.at[brier_df.index[i],"time"], brier_df.at[brier_df.index[i], "pred_alpha"], brier_df.at[brier_df.index[i], "pred_beta"]) 
        # add contribution to Brier score
        brier_score += S_i**2 * brier_df.at[brier_df.index[i], "dead_by_t"] 
        brier_score += (1-S_i)**2 * brier_df.at[brier_df.index[i], "survive_t"] 

    brier_score /= N 

    return brier_score

"""
Input:
    - test_result : Pandas dataframe of time, status, pred_alpha, pred_beta 
    - num_points : number of points in the partition used for numerical integration (Simpson's rule)
Output:
    - the (approx.) integrated Brier score (with t_1=0, t_2=max(time))
"""

def integrated_brier(test_result, num_points=100):

    # partition [0,T] into num_points evenly spaced sub-intervals
    T = test_result["time"].max()
    t_vals = np.linspace(0,T,num=num_points)
    # compute BS(t) as each point
    brier_scores = [brier(test_result, t) for t in t_vals]
    # integrated (by Simpson's rule) and divide by T (i.e. end time - start time) 
    int_brier_score = simps(brier_scores, t_vals)/T

    return int_brier_score

