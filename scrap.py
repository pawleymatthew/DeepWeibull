import numpy as np
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

t_vals =np.array([1,2,3,5.5])
t_max = t_vals.max()

print(max(math.ceil(t_max),5))
