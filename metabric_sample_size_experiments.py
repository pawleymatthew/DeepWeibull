import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from data import make_train_test
from data import normalise
from evaluation_metrics import c_index
from regression_weibull import regression_weibull
from deep_weibull import deep_weibull
from deep_hit import deep_hit
from deep_hit_zero_alpha import deep_hit_zero_alpha
from pycox.datasets import metabric

models=["regression_weibull","deep_weibull","deep_hit"]

"""
data = {
    'train_frac': [],
    'model':[],
	'mean_c': []}
results_df = pd.DataFrame(data)
results_df.to_csv("metabric_size_results.csv")
"""

results_df = pd.read_csv("metabric_size_results.csv")

for train_frac in np.arange(0.60, 0.70, 0.05).tolist():

    df = metabric.read_df() # read in dataset
    df = normalise(df, ['x0', 'x1', 'x2', 'x3', 'x8']) # normalise cols where appropriate
    df.rename(columns={"duration": "time", "event": "status"}, inplace=True) # rename duration/event cols
    make_train_test(df, train_frac=train_frac, dataset="metabric_temp", n_splits=10) # make train/test splits

    for model in models:

        c_vals = [None] * 10

        for split in [1,2,3,4,5,6,7,8,9,10]:

            # run the model
            f_model = globals()[model]
            result = f_model("metabric_temp", split)

            # get evaluation object
            ev = result["ev"]
            # compute c index
            c_vals[split-1] = c_index(ev)

        c = np.mean(c_vals)
        c = round(c, ndigits=3)

        new_row = {'train_frac':train_frac, 'model':model, 'mean_c':c}
        results_df = results_df.append(new_row, ignore_index=True)


results_df.to_csv("metabric_size_results.csv")

fig = plt.figure()
ax = plt.axes()

results_rw = results_df[results_df["model"]=="regression_weibull"]
results_dw = results_df[results_df["model"]=="deep_weibull"]
results_dh = results_df[results_df["model"]=="deep_hit"]

l1 = ax.plot(results_rw["train_frac"], results_rw["mean_c"], color="blue")[0]
l2 = ax.plot(results_dw["train_frac"], results_dw["mean_c"], color="orange")[0]
l3 = ax.plot(results_dh["train_frac"], results_dh["mean_c"], color="green")[0]

ax.legend([l1, l3, l3],     # The line objects
           labels=["RegressionWeibull","DeepWeibull","DeepHit"]  # The labels for each line
           )

plt.xlabel('Proportion of individuals in training set (%)')
plt.ylabel(r'$c$'+"-index")

plot_file_path = "plots/real_data_experiments/metabric_training_size.pdf"
plt.savefig(plot_file_path)
