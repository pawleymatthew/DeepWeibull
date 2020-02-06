import numpy as np
import pandas as pd
import math
import matplotlib as plt

from deep_hit import deep_hit
from deep_weibull import deep_weibull
from regression_weibull import regression_weibull
from evaluation_metrics import c_index, brier_score

datasets = ["small_synthetic_weibull", "metabric", "support","rr_nl_nhp"]
models = ["regression_weibull", "deep_hit", "deep_hit_zero", "deep_weibull"]

# For each dataset, run the models, compute the evaluation metrics, and create plots

c_index_df = pd.DataFrame(columns=datasets, index=models)
c_index_df["model"] = models 

int_brier_df = pd.DataFrame(columns=datasets, index=models)
int_brier_df["model"] = models

for dataset in datasets:

    # Run the models
    rw = regression_weibull(dataset)
    dh = deep_hit(dataset, alpha=0.3, lr=0.01, epochs=50, batch_size=256)
    dh_zero = deep_hit(dataset, alpha=0.0, lr=0.01, epochs=50, batch_size=256)
    dw = deep_weibull(dataset, lr=0.01, epochs=50, steps_per_epoch=10)

    # Compute c_index 
    rw_c = c_index(rw["ev"])
    dh_c = c_index(dh["ev"])
    dh_zero_c = c_index(dh_zero["ev"])
    dw_c = c_index(dw["ev"])

    c_index_df[dataset] = [rw_c, dh_c, dh_zero_c, dw_c]
    c_index_df.to_csv(r"evaluation_metrics_results/c_index.csv", index=False)

    # Compute integrated Brier scores
    rw_ib = brier_score(rw["ev"], dataset)["int_score"]
    dh_ib = brier_score(dh["ev"], dataset)["int_score"]
    dh_zero_ib = brier_score(dh_zero["ev"],dataset)["int_score"]
    dw_ib = brier_score(dw["ev"],dataset)["int_score"]

    int_brier_df[dataset] = [rw_ib, dh_ib, dh_zero_ib, dw_ib]
    int_brier_df.to_csv(r"evaluation_metrics_results/int_brier_score.csv", index=False)

    



