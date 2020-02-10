import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from regression_weibull import regression_weibull
from deep_weibull import deep_weibull
from deep_hit import deep_hit
from deep_hit_zero_alpha import deep_hit_zero_alpha

from evaluation_metrics import c_index, brier_score

def run_models(datasets, models):

    metrics_df = pd.read_pickle("evaluation_metrics_results/c_index_and_int_brier.pkl")
    
    for dataset in datasets:

        # path to the Brier scores
        brier_scores_df_path = "evaluation_metrics_results/brier_scores/" + dataset + ".pkl"
        brier_scores_df = pd.read_pickle(brier_scores_df_path)

        for model in models:

            # run the model
            f_model = globals()[model]
            result = f_model(dataset)

            # get evaluation object
            ev = result["ev"]
            # compute c index
            c = c_index(ev)
            # compute brier score (scores + integrated)
            brier_object = brier_score(ev, dataset)
            int_score = brier_object["int_score"]

            # add the results to the files
            metrics_df.loc[dataset, model] = [c, int_score]
            brier_scores_df["t"] = list(brier_object["scores"].index)
            brier_scores_df[model] = list(brier_object["scores"].values)

            # write the the test result dataframe to a file
            test_result_file_path = "test_results/" + model + "/" + dataset + ".pkl"
            result["test_result"].to_pickle(test_result_file_path)

        # save brier scores results for this dataset
        brier_scores_df.to_pickle(brier_scores_df_path)
            
    print(metrics_df)
    metrics_df.to_pickle("evaluation_metrics_results/c_index_and_int_brier.pkl")


models = ["regression_weibull","deep_weibull","deep_hit", "deep_hit_zero_alpha"]
models = ["regression_weibull"]
datasets = ["small_synthetic_weibull", "big_synthetic_weibull", "metabric", "support", "rr_nl_nhp"]
run_models(datasets, models)

"""
Create/update plots of all Brier scores dataframes
This updates the plots for all datasets/models (irrespective of which models/datasets were input in run_models())
"""

# create tidy strings of names for use in plots
tidy_models = {
  "regression_weibull": "Regression Weibull",
  "deep_weibull": "Deep Weibull",
  "deep_hit": "Deep Hit",
  "deep_hit_zero_alpha": "Deep Hit ($\\alpha =0$)"
}

tidy_datasets = {
  "small_synthetic_weibull": "Small Synthetic Weibull",
  "big_synthetic_weibull": "Big Synthetic Weibull",
  "metabric": "METABRIC",
  "support": "SUPPORT",
  "rr_nl_nhp": "RRNLNHP"
}

for dataset in ["small_synthetic_weibull", "big_synthetic_weibull", "metabric", "support", "rr_nl_nhp"]:

    brier_scores_df_path = "evaluation_metrics_results/brier_scores/" + dataset + ".pkl"
    brier_scores_df = pd.read_pickle(brier_scores_df_path)

    # path for saving Brier score plots
    brier_score_plot_path = "plots/brier_scores/" + dataset + ".png"

    # create plot of the brier scores for this dataset
    brier_scores_df.plot(x=brier_scores_df.columns[0], y=brier_scores_df.columns[1:])
    plt.xlabel("Time")
    plt.ylabel("Brier Score")
    plt.title("Brier Score: " + tidy_datasets[dataset])
    plt.legend(tidy_models.values())
    plt.savefig(brier_score_plot_path)
    plt.clf()

