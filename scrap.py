import pandas as pd
import numpy as np

file_path = "test_results/regression_weibull/support.pkl"

df = pd.read_pickle(file_path)

print(df.describe())

a = [1,2,3,4,5,6]
b = [4,5,1,8,9,1]

print(np.minimum(a,b))