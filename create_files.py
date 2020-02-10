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

write_file_path = "evaluation_metrics_results/c_index_and_int_brier.pkl"

iterables = [datasets, models]
index = pd.MultiIndex.from_product(iterables, names=['dataset', 'model'])
metrics_df = pd.DataFrame(index=index)
metrics_df["c_index"] = pd.Series()
metrics_df["int_brier_score"] = pd.Series()

metrics_df.to_pickle(write_file_path)

"""
Create (pickle) file to store Brier score values (one file for each dataset)
"""

for dataset in datasets:

    write_file_path = "evaluation_metrics_results/brier_scores/" + dataset + ".pkl"
    
    colnames = ['t'] + models
    df = pd.DataFrame(columns=colnames)
    
    df.to_pickle(write_file_path)
