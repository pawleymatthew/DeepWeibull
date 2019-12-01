import pandas as pd
import numpy as np
from itertools import product
from performance_metrics import time_dependent_concordance

#models = ["deep_weibull", "simple_model_one", "simple_model_two"]
models = ["deep_weibull"]
datasets = ["linear_weibull", "non_linear_weibull"]

performance_df = pd.DataFrame(list(product(models, datasets)), columns=['model', 'dataset'])
c_index = []

for model_name in models:
    for dataset_name in datasets:
        results_path = "predict_results/" + model_name + "_~_" + dataset_name + "_results.csv"
        test_result = pd.read_csv(results_path)
        c = time_dependent_concordance(test_result, sample_frac=0.1)
        c_index.append(c)

performance_df["c_index"] = c_index

print(performance_df)

