import matplotlib.pyplot as plt
import numpy as np
import math


"""
Create plot of Exponential and Weibull distribution hazard/survival curves.
"""

plot_file_path = "plots/parametric_dist_curves/weibull_dist_curves.pdf" # path to output file

S = lambda t,a,b: np.exp(-(t/a)**b) # survival function of Weibull(a,b)
h = lambda t,a,b: (b/a)*(t/a)**(b-1) # hazard function of Weibull(a,b)

# t values
t = np.arange(start=0.001,stop=100.001,step=0.2) 
# fixed alpha
alpha = 50 
# beta values
beta1 = 0.75 
beta2 = 1
beta3 = 3

# survival curves
S1 = S(t, alpha, beta1)
S2 = S(t, alpha, beta2)
S3 = S(t, alpha, beta3)
# hazard curves
h1 = h(t, alpha, beta1)
h2 = h(t, alpha, beta2)
h3 = h(t, alpha, beta3)

# set up fig
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,12.5)) # 1x2 subplots
# fig.suptitle(r'Weibull distribution ($\alpha=50$)') # main title
line_labels = [r'$\alpha=50,\beta=0.75$', r'$\alpha=50,\beta=1$', r'$\alpha=50,\beta=3$'] # labels for legend
# create subplots
l1 = ax1.plot(t, S1, color="blue")[0]
l2 = ax1.plot(t, S2, color="orange")[0]
l3 = ax1.plot(t, S3, color="green")[0]
l4 = ax2.plot(t, h1, color="blue")[0]
l5 = ax2.plot(t, h2, color="orange")[0]
l6 = ax2.plot(t, h3, color="green")[0]
# create legend
fig.legend([l1, l2, l3, l4, l5, l6],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="lower center",   # Position of legend
           borderaxespad=0.0,
           title="Weibull parameters"  # Title for the legend
           )
# axis labels
ax1.set_xlabel(r'$t$')
ax2.set_xlabel(r'$t$')
ax1.set_ylabel(r'$S(t)$')
ax2.set_ylabel(r'$\mu(t)$')
ax1.set_title("Survival function")
ax2.set_title("Hazard function")
# adjust and save fig
#plt.subplots_adjust(best)
ax1.set_aspect('auto')
ax2.set_aspect('auto')
plt.savefig(plot_file_path)

