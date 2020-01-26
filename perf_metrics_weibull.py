import numpy as np
import pandas as pd

from scipy.integrate import simps

from weibull_dist import weibull_survival, weibull_int_hazard


"""
TO DO: ADD CENSORING DISTRIBUTION TO THE PERFORMANCE METRICS SO THEY ARE COMPARABLE WITH THE DEEP HIT ONES!!!!
"""


def c_td_weibull(test_result):

    """
    DESCRIPTION: 
        Computes the time-dependent concordance index given a set of survival data and corresponding Weibull models.
    INPUT:
        - test_result: a Pandas dataframe with columns [time, status, pred_alpha, pred_beta]
    OUTPUT:
        - the time-dependent concordance index
    """

    test_result.sort_values(by="time", ascending=True, inplace=True) # order the individuals by time column
    N = test_result.shape[0] # number of individuals

    # initialise counts
    comp = 0 # number of comparable pairs (no. of edges E_{ij} in order graph)
    conc = 0 # number of concordant pairs 

    for i in range(N-1): # for all but last individual

        if test_result.at[test_result.index[i], "status"] == 1: # if ind. i uncensored

            # get death time and predicted model for ind. i
            t_i = test_result.at[test_result.index[i], "time"]
            alpha_i = test_result.at[test_result.index[i], "pred_alpha"]
            beta_i = test_result.at[test_result.index[i], "pred_beta"]
            # store M(t_i|x_i)
            M_ii = weibull_int_hazard(t_i, alpha_i, beta_i) 

            for j in range(i+1,N): # for inds. who (certainly) survived longer than ind. i
                    
                t_j = test_result.at[test_result.index[j], "time"] # get death/censoring time
                    
                if t_i < t_j: # check times are not tied

                    comp += 1 # (i,j) are comparable pair

                    # get predicted model for ind. j
                    alpha_j = test_result.at[test_result.index[j], "pred_alpha"]
                    beta_j = test_result.at[test_result.index[j], "pred_beta"]

                    # check if M(t_i|x_i) > M(t_i|x_j) (i.e. S(t_i|x_i) < S(t_i|x_j))
                    if M_ii > weibull_int_hazard(t_i, alpha_j, beta_j):

                        conc += 1 # (i,j) are concordant

    # c-index = prop. of comparable pairs that are concordant
    return float(conc/comp)


def brier_weibull(test_result, t):

    """
    DESCRIPTION: 
        Computes the Brier score w.r.t a fixed timepoint t given a set of survival data and corresponding Weibull models.
    INPUT:
        - test_result: a Pandas dataframe with columns [time, status, pred_alpha, pred_beta]
    OUTPUT:
        - the Brier score BS(t) w.r.t to timepoint t
    """

    # indicator column: dead by time t (i.e. time < t, status = uncensored)
    test_result["dead_by_t"] = np.where((test_result["time"]<=t) & (test_result["status"]==1), 1, 0)
    # indicator column: survive to at least time t (i.e. time > t)
    test_result["survive_t"] = np.where(test_result["time"]>t, 1, 0)

    b = 0 # initialise Brier score
    N = test_result.shape[0] # number of individuals

    for i in range(N): # for each individual i
        row_i = test_result.index[i] # store index of individual i
        # compute S(t_i|x_i)
        S_i = weibull_survival(test_result.at[row_i,"time"], test_result.at[row_i,"pred_alpha"], test_result.at[row_i,"pred_beta"]) 
        # add contribution to Brier score
        b += S_i**2 * test_result.at[row_i,"dead_by_t"] 
        b += (1-S_i)**2 * test_result.at[row_i,"survive_t"] 
 
    b /= N # divide b by N
    
    return b


def int_brier_weibull(test_result, int_points=100):

    """
    DESCRIPTION: 
        Computes the integrated Brier score given a set of survival data and corresponding Weibull models. 
        The integration is performed numerically using Simpson's rule.
        The number of points in the partition should be sufficiently large for a good estimate.
        The timepoints are taken between the min and max of the "time" column.
        The (t,BS(t)) pairs are also an output, so a plot.
    INPUT:
        - test_result: a Pandas dataframe with columns [time, status, pred_alpha, pred_beta]
        - int_points : the number of timepoints t for the numerical integration.
    OUTPUT:
        - int_brier : the integrated Brier score
        - t_vals : the timepoints using for the computation
        - brier_t_vals : the BS(t) scores for each t in t_vals
    """

    # create partition of timepoints to compute BS(t) over
    T_min = test_result["time"].min() # min. time
    T_max = test_result["time"].max() # max. time
    t_vals = np.linspace(T_min,T_max,num=int_points) # partition [T_min,T_max] with int_points sub-intervals

    # compute BS(t) as each point
    b_t = [brier_weibull(test_result, t) for t in t_vals]
    # integrate (by Simpson's rule) and divide by total width of time interval (i.e. T_max - T_min)
    int_b = float(simps(b_t, t_vals)/(T_max - T_min))

    return ({
            "int_brier" : int_b,
            "t_vals" : t_vals,
            "brier_t_vals" : b_t})


"""
ORIGINAL CODE FOR BRIER_WEIBULL(t)
I MAKE A COPY OF THE TEST RESULT DF - IS IT NECESSARY?
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
"""

