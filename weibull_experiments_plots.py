import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


"""
1. Plot the survival curves of a patient i output by each of the four models.
2. For a patient i and a time t, plot ("heat map") the predicted hazard as a function of x_1,x_2 for Actual/DWeibull/RWeibull.
"""

"""
Plot the survival curves of a patient i output by each of the four models.
"""

def plot_survival_curves(dataset="big_nonlinear_weibull", split=1):
    
    # weibull survival function
    S = lambda t,a,b: np.exp(-1*np.power(np.divide(t,a),b)) # survival function of Weibull(a,b)

    # path to output file
    plot_file_path = "plots/weibull_experiments/survival_curves/" + dataset + "_" + str(split) + ".pdf" 

    """
    Get the predicted alpha/beta values from RegWeib
    """
    #rw_test_path = "test_results/regression_weibull/" + dataset + "_" + str(split) + ".pkl"
    #rw_test_df = pd.read_pickle(rw_test_path)
    #rw_pred_alpha = rw_test_df.loc[i,"pred_alpha"]
    #rw_pred_beta = rw_test_df.loc[i,"pred_beta"]
   
    """
    Get the predicted alpha/beta values from DeepWeib
    """
    #dw_test_path = "test_results/deep_weibull/" + dataset + "_" + str(split) + ".pkl"
    #dw_test_df = pd.read_pickle(dw_test_path)
    #dw_pred_alpha = dw_test_df.loc[i,"pred_alpha"]
    #dw_pred_beta = dw_test_df.loc[i,"pred_beta"]


    # get the actual parameter values
    #fun = lambda x :  80 - 40*x[0]**2 + 30*x[0]*x[1]
    #x = dw_test_df.loc[i,["x0","x1"]]
    #actual_alpha = fun(x)
    #actual_beta = 1.1

    """ 
    Get the predicted S(t) values for DeepHit
    """
    #dh_test_path = "test_results/deep_hit/" + dataset + "_" + str(split) + ".pkl"
    #dh_test_df = pd.read_pickle(dh_test_path)
    #dh_s = dh_test_df.iloc[:,i]
    #dh_t_vals = dh_test_df.index.values.tolist() # list of the t values

    # compute the survival curves at t_vals

    weibull_t_vals = np.arange(start=0.001,stop=500,step=0.2) 
    dh_t_vals = np.arange(start=0.001,stop=500,step=1.0)

    rw_pred_alpha = 77.032
    rw_pred_beta = 1.237

    dw_pred_alpha = 38.611
    dw_pred_beta = 1.044

    dh_pred_alpha = 42.183
    dh_pred_beta = 1.168

    actual_alpha = 35.595
    actual_beta = 1.10

    S_rw = S(weibull_t_vals, rw_pred_alpha, rw_pred_beta)
    S_dw = S(weibull_t_vals, dw_pred_alpha, dw_pred_beta)
    S_dh = S(dh_t_vals, dh_pred_alpha, dh_pred_beta)
    S_actual = S(weibull_t_vals, actual_alpha, actual_beta)

    plt.figure()
    l_rw = plt.plot(weibull_t_vals, S_rw, color="blue")
    l_dw = plt.plot(weibull_t_vals, S_dw, color="orange")
    l_dh = plt.plot(dh_t_vals, S_dh, color="green")
    l_actual = plt.plot(weibull_t_vals, S_actual, color="black",linestyle="dashed")

    line_labels = ["RegressionWeibull", "DeepWeibull", "DeepHit", "Actual"] # labels for legend
    plt.legend([l_rw, l_dw, l_dh, l_actual],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper right",   # Position of legend
           borderaxespad=0.2, # Title for the legend
           )
    
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\hat{S}(t)$')
    plt.savefig(plot_file_path)

#plot_survival_curves() 

"""
Plot the hazard 'map' at a fixed time
"""

# make these smaller to increase the resolution
dx, dy = 0.01, 0.01

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(-1, 1 + dy, dy),
                slice(-1, 1 + dx, dx)]


t=[5]

actual_beta = np.array([1.10])
actual_mu = (actual_beta) * t**(actual_beta-1) * np.power(80 - 40*np.square(x) + 30*np.multiply(x,y),-1*actual_beta)

rw_pred_beta = np.array([1.237])
rw_pred_mu = (actual_beta) * t**(actual_beta-1) * np.power(77.2 - 0.120*x - 0.096*y,-1*actual_beta)


# contours are *point* based plots, so convert our bound into point
# centers
fig, (ax0, ax1) = plt.subplots(nrows=2)

cmap = plt.get_cmap('PiYG')

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
actual_mu = actual_mu[:-1, :-1]
rw_pred_mu = rw_pred_mu[:-1, :-1]
levels_actual = MaxNLocator(nbins=15).tick_values(actual_mu.min(), actual_mu.max())
levels_rw_pred = MaxNLocator(nbins=15).tick_values(rw_pred_mu.min(), rw_pred_mu.max())

actual_cf = ax0.contourf(x[:-1, :-1] + dx/2.,
                  y[:-1, :-1] + dy/2., actual_mu, levels=levels_actual,
                  cmap=cmap)
fig.colorbar(actual_cf, ax=ax0)
ax0.set_title('Actual')

rw_pred_cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                  y[:-1, :-1] + dy/2., rw_pred_mu, levels=levels_rw_pred,
                  cmap=cmap)
fig.colorbar(rw_pred_cf, ax=ax1)
ax1.set_title('RegressionWeibull')

ax0.set_xlabel(r'$x_1$')
ax1.set_xlabel(r'$x_1$')
ax0.set_ylabel(r'$x_2$')
ax1.set_ylabel(r'$x_2$')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plot_file_path = "plots/weibull_experiments/hazard_maps/hazard_map.pdf" 
plt.savefig(plot_file_path)
