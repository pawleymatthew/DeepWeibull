import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from regression_weibull import regression_weibull
from deep_weibull import deep_weibull
from deep_hit import deep_hit
from deep_hit_zero_alpha import deep_hit_zero_alpha

from evaluation_metrics import c_index, brier_score

# create tidy strings of names for use in plots/tables
tidy_models = {
  "regression_weibull": "Regression Weibull",
  "deep_weibull": "Deep Weibull",
  "deep_hit": "Deep Hit",
  "deep_hit_zero_alpha": "Deep Hit ($\\alpha =0$)"
}

tidy_datasets = {
  "small_linear_weibull": "Small Linear Weibull",
  "big_linear_weibull": "Large Linear Weibull",
  "small_nonlinear_weibull": "Small Linear Weibull",
  "big_nonlinear_weibull": "Large Non-Linear Weibull",
  "metabric": "METABRIC",
  "support": "SUPPORT",
  "rrnlnph": "RRNLNPH"
}

def run_models(datasets, splits, models):

    metrics_df = pd.read_pickle("evaluation_metrics_results/c_index_and_int_brier.pkl")
    
    for split in splits:

        for dataset in datasets:

            # path to the Brier scores
            brier_scores_df_path = "evaluation_metrics_results/brier_scores/" + dataset + "_" + str(split) + ".pkl"
            brier_scores_df = pd.read_pickle(brier_scores_df_path)

            for model in models:

                # run the model
                f_model = globals()[model]
                result = f_model(dataset, split)

                # get evaluation object
                ev = result["ev"]
                # compute c index
                c = c_index(ev)
                # compute brier score (scores + integrated)
                brier_object = brier_score(ev, dataset, split)
                int_score = brier_object["int_score"]

                # add the results to the files
                metrics_df.loc[(dataset, split, model), 'c_index'] = c
                metrics_df.loc[(dataset, split, model), 'int_brier_score'] = int_score
                brier_scores_df["t"] = list(brier_object["scores"].index)
                brier_scores_df[model] = list(brier_object["scores"].values)

                # write the the test result dataframe to a file
                test_result_file_path = "test_results/" + model + "/" + dataset + "_" + str(split) + ".pkl"
                result["test_result"].to_pickle(test_result_file_path)

            # save brier scores results for this dataset and split
            brier_scores_df.to_pickle(brier_scores_df_path)
            
    print(metrics_df)
    metrics_df.to_pickle("evaluation_metrics_results/c_index_and_int_brier.pkl")

    # write LaTeX tables
    #c_index_df = metrics_df["c_index"].unstack(level=0)
    #c_index_df = c_index_df.round(3)
    #int_brier_score_df = metrics_df["int_brier_score"].unstack(level=0)
    #int_brier_score_df = int_brier_score_df.round(4)
    #print(c_index_df.to_latex())
    #print(int_brier_score_df.to_latex())

"""
RUN THE MODELS
"""

"""
WEIBULL EXPERIMENTS
"""

#models= ["deep_hit_zero_alpha"]
#splits = [1,2,3]
#datasets = ["small_linear_weibull","big_linear_weibull","small_nonlinear_weibull","big_nonlinear_weibull"]
#run_models(datasets, splits, models)

"""
REAL DATA EXPERIMENTS
"""

models = ["deep_hit_zero_alpha"]
splits = [1,2,3]
datasets = ["support"]
run_models(datasets, splits, models)


"""
Create/update plots of all Brier scores dataframes
This updates the plots for all datasets/models (irrespective of which models/datasets were input in run_models())
"""

for dataset in ["small_linear_weibull", "big_linear_weibull", "small_nonlinear_weibull", "big_nonlinear_weibull", "metabric", "support", "rrnlnph"]:

    for split in range(1,4):

        brier_scores_df_path = "evaluation_metrics_results/brier_scores/" + dataset + "_" + str(split) + ".pkl"
        brier_scores_df = pd.read_pickle(brier_scores_df_path)

        # path for saving Brier score plots
        brier_score_plot_path = "plots/brier_scores/" + dataset + "/" + dataset + "_" + str(split) + ".pdf"

        # create plot of the brier scores for this dataset and split
        brier_scores_df.plot(x=brier_scores_df.columns[0], y=brier_scores_df.columns[1:])
        plt.xlabel("Time")
        plt.ylabel("Brier Score")
        plt.title("Brier Score: " + tidy_datasets[dataset] + " (Split " + str(split)+ ")")
        plt.legend(tidy_models.values())
        plt.savefig(brier_score_plot_path)
        plt.clf()
        plt.close('all')

