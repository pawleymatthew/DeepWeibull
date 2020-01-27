import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from perf_metrics_weibull import c_td_weibull, int_brier_weibull
from perf_metrics_deephit import c_td_deep_hit, int_brier_deep_hit

models = ["simple_model_one", "simple_model_two", "deep_weibull", "deep_hit", "deep_hit_zero_alpha"]
datasets = ["linear_weibull","non_linear_weibull"]


performance_df = pd.DataFrame(list(product(datasets, models)), columns=['dataset', 'model'])
c_index = []
int_brier_score = []

for dataset_name in datasets:

    for model_name in models:
        
        results_path = "predict_results/" + model_name + "_~_" + dataset_name + "_results.csv"
        test_result = pd.read_csv(results_path)

        if model_name in ["deep_hit_zero_alpha", "deep_hit"]:

            c = c_td_deep_hit(test_result) # compute c-index
            c_index.append(c) # add to list of results

            int_brier_object = int_brier_deep_hit(test_result) # compute integrated Brier score (plus the (t, BS(t)) values)
            int_brier_score.append(int_brier_object["int_brier"]) # add integrated Brier score to list of results

        else:

            c = c_td_weibull(test_result) # compute c-index
            c_index.append(c) # add to list of results

            int_brier_object = int_brier_weibull(test_result) # compute integrated Brier score (plus the (t, BS(t)) values)
            int_brier_score.append(int_brier_object["int_brier"]) # add integrated Brier score to list of results

            # make plot of (t,BS(t)) here...

        plt.plot(int_brier_object["t_vals"], int_brier_object["brier_t_vals"], label=model_name)   
            
            
    plt.xlabel("t")
    plt.ylabel("BS(t)")
    plt.title('Brier score: '+ dataset_name)
    plt.legend()
    plt.savefig("brier_plot" + "_" + dataset_name + ".png")
    plt.clf()

performance_df["c_index"] = c_index
performance_df["int_brier_score"] = int_brier_score

performance_df.to_csv("performance_results.csv", index=False)

print(performance_df)

