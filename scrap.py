import pandas as pd
import numpy as np


# TEST RESULTS
df_rw = pd.read_pickle(r"test_results/regression_weibull/support_1.pkl")
df_dw = pd.read_pickle(r"test_results/deep_weibull/support_1.pkl")
#df_dh = pd.read_pickle(r"test_results/deep_hit/support_1.pkl")
print(df_rw.iloc[1])
print(max(df_dw["pred_beta"]))

# BRIER SCORES
#df = pd.read_pickle(r"evaluation_metrics_results/brier_scores/metabric_1.pkl")

# C-INDEX AND INTEGRATED BRIER SCORES
#df = pd.read_pickle(r"evaluation_metrics_results/c_index_and_int_brier.pkl")
#print(df.iloc[60:72])
