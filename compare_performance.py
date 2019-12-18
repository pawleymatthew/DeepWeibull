import pandas as pd
import numpy as np
from itertools import product
from performance_metrics import time_dependent_concordance, brier

models = ["deep_weibull", "simple_model_one", "simple_model_two"]
datasets = ["linear_weibull", "non_linear_weibull","metabric"]

models = ["deep_weibull"]
datasets = ["linear_weibull", "non_linear_weibull"]

performance_df = pd.DataFrame(list(product(models, datasets)), columns=['model', 'dataset'])
c_index = []
brier_score = []

for model_name in models:
    for dataset_name in datasets:
        results_path = "predict_results/" + model_name + "_~_" + dataset_name + "_results.csv"
        test_result = pd.read_csv(results_path)
        
        # time-dependent concordance index
        c = time_dependent_concordance(test_result)
        c_index.append(c)

        # integrated brier score
        b = brier(test_result, t=40)
        brier_score.append(b)

performance_df["c_index"] = c_index
performance_df["brier_score"] = brier_score

print(performance_df)

