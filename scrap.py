import pandas as pd
from pycox.evaluation import EvalSurv
import numpy as np
from perf_metrics_weibull import c_td_weibull, int_brier_weibull
from perf_metrics_deephit import deep_hit_metrics_preprocess, int_brier_deep_hit, c_td_deep_hit

results_path = "predict_results/" + "deep_hit" + "_~_" + "non_linear_weibull" + "_results.csv"
test_result = pd.read_csv(results_path)

"""
print(test_result.head)
print(test_result["time"])

test_time = np.array(test_result.pop("time"))
test_status = np.array(test_result.pop("status"))
surv = test_result.copy()
surv = surv.transpose()
surv = surv.reset_index(drop=True)
deep_hit_eval_surv = EvalSurv(surv, test_time, test_status, censor_surv='km')
c = deep_hit_eval_surv.concordance_td('antolini')
print(c)

test_result = pd.read_csv(results_path)
T_min = test_result["time"].min()
T_max = test_result["time"].max()

int_briers = deep_hit_eval_surv.integrated_brier_score(np.linspace(T_min,T_max,num=20))
print(int_briers)
"""

print(min(1,2))


