import numpy as np
import pandas as pd
import math
import matplotlib as plt

from deep_hit import deep_hit
from deep_weibull import deep_weibull
from regression_weibull import regression_weibull
from evaluation_metrics import c_index, brier_score

datasets = ["small_synthetic_weibull", "metabric", "support"]
models = ["regression_weibull", "deep_hit", "deep_hit_zero", "deep_weibull"]

# For each dataset, run the models, compute the evaluation metrics, and create plots


for dataset in datasets:

    # Run the models
    rw = regression_weibull(dataset)
    dh = deep_hit(dataset, alpha=0.3, lr=0.01, epochs=50, batch_size=256)
    dh_zero = deep_hit(dataset, alpha=0.0, lr=0.01, epochs=50, batch_size=256)
    dw = deep_weibull(dataset, lr=0.01, epochs=50, steps_per_epoch=2)






