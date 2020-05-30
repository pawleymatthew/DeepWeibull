import numpy as np
import pandas as pd
import math

from itertools import product

models = ["regression_weibull", "deep_weibull", "deep_hit", "deep_hit_zero_alpha"]
datasets = ["small_synthetic_weibull", "big_synthetic_weibull", "metabric", "support", "rr_nl_nhp"]
metrics = ["c_index", "int_brier_score"]

"""
Create (pickle) file to store c-index and integrated Brier score results (all models and datasets in one file)
"""

datasets = ["small_linear_weibull", "big_linear_weibull", "small_nonlinear_weibull", "big_nonlinear_weibull", "metabric", "support", "rrnlnph"]
splits = range(1,4)
models = ["regression_weibull", "deep_weibull", "deep_hit", "deep_hit_zero_alpha"]
metrics_df_index = pd.MultiIndex.from_product([datasets, splits, models], names=["dataset", "split", "model"])
metrics_df = pd.DataFrame(columns=["c_index", "int_brier_score"], index=metrics_df_index)
metrics_df.to_pickle("evaluation_metrics_results/c_index_and_int_brier.pkl")

"""
Create (pickle) file to store Brier score values (one file for each dataset)
"""

for dataset in datasets:
    for split in splits:

        write_file_path = "evaluation_metrics_results/brier_scores/" + dataset + "_" + str(split) + ".pkl"
        
        colnames = ['t'] + models
        df = pd.DataFrame(columns=colnames)
        
        df.to_pickle(write_file_path)
