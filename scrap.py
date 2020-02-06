import numpy as np
import pandas as pd
import math
from deep_hit import deep_hit
from deep_weibull import deep_weibull
from regression_weibull import regression_weibull

from evaluation_metrics import c_index, brier_score

"""
dataset = "metabric"
run = regression_weibull(dataset)

c = c_index(run["ev"])
b = brier_score(run["ev"], dataset)

print(c)
print(b["scores"])
print(b["int_score"])
"""

datasets = ["small_synthetic_weibull", "metabric", "support"]
models = ["regression_weibull", "deep_hit", "deep_hit_zero", "deep_weibull"]

# For each dataset, run the models, compute the evaluation metrics, and create plots

c_index_df = pd.DataFrame(columns=datasets, index=models)

print(c_index_df)
