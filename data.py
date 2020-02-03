import pandas as pd
import numpy as np
import random

from pycox.datasets import metabric, support, flchain, rr_nl_nhp

np.random.seed(1234)

"""
PREPROCESSING FUNCTIONS
"""

def normalise(df, colnames):
    df[colnames] = df[colnames].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    return df

    
def make_train_test(df, train_frac):
    train_df = df.groupby("status").apply(lambda x: x.sample(frac=train_frac)) # censoring frac. equal in train and test sets
    train_df = train_df.reset_index(level="status", drop=True)
    train_df = train_df.sort_index()
    test_df = df.drop(train_df.index)
    return ({"train_df" : train_df, "test_df" : test_df})


"""
I use one simulated and two real-world datasets from "pycox.datasets". 
For further details see the "pycox" documentation at https://github.com/havakv/pycox#references.

- metabric [2]: (1,904 obs). The Molecular Taxonomy of Breast Cancer International Consortium (METABRIC).
- support [2]: (8,873 obs). Study to Understand Prognoses Preferences Outcomes and Risks of Treatment (SUPPORT). 
- rr_nl_nhp [1]: (25,000 obs.) This is a continuous-time simulation study with event times drawn from a relative risk non-linear non-proportional hazards model (RRNLNPH).

[1] Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. Journal of Machine Learning Research, 20(129):1–30, 2019. 
[2] Jared L. Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology, 18(1), 2018. 
"""


"""
METABRIC
    - Columns 'x0', 'x1', 'x2', 'x3', 'x8' need to be normalised.
    - Columns 'x4', 'x5', 'x6', 'x7' are binary, leave these alone.
    - Change ['duration', 'event'] -> ['time', 'status'].
    - Train/test split of 80/20.
"""

df = metabric.read_df() # read in dataset
df = normalise(df, ['x0', 'x1', 'x2', 'x3', 'x8']) # normalise cols where appropriate
df.rename(columns={"duration": "time", "event": "status"}, inplace=True) # rename duration/event cols
sets = make_train_test(df, 0.8) # make train/test sets

df.to_csv(r"datasets/metabric_data/metabric_df.csv", index=False)
sets["train_df"].to_csv(r"datasets/metabric_data/metabric_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/metabric_data/metabric_test_df.csv", index=False)

"""
SUPPORT
    - Columns 'x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13' need to be normalised.
    - Columns 'x1', 'x4', 'x5' are binary, leave these alone.
    - Columns 'x2', 'x3', 'x6' are categorical. I have removed these (they were 'race', 'number of comorbidities' and 'presence of cancer')
    - Change ['duration', 'event'] -> ['time', 'status'].
    - Train/test split of 80/20.
"""

df = support.read_df() # read in dataset
df = normalise(df, ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']) # normalise cols where appropriate
df = df.drop(['x2', 'x3', 'x6'], axis=1)
df.rename(columns={"duration": "time", "event": "status"}, inplace=True) # rename duration/event cols
sets = make_train_test(df, 0.9) # make train/test sets

df.to_csv(r"datasets/support_data/support_df.csv", index=False)
sets["train_df"].to_csv(r"datasets/support_data/support_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/support_data/support_test_df.csv", index=False)

"""
RR_NL_NHP
    - No columns need to be normalised.
    - Columns 'duration_true', 'event_true', 'censoring_true' contain extraneous intermediate information about the simulation.
    - Change ['duration', 'event'] -> ['time', 'status'].
    - Train/test split of 80/20.
"""

df = rr_nl_nhp.read_df() # read in dataset
df = df.drop(['duration_true', 'event_true', 'censoring_true'], axis=1) # remove extraneous cols
df.rename(columns={"duration": "time", "event": "status"}, inplace=True) # rename duration/event cols
sets = make_train_test(df, 0.8) # make train/test sets

df.to_csv(r"datasets/rr_nl_nhp_data/rr_nl_nhp_df.csv", index=False)
sets["train_df"].to_csv(r"datasets/rr_nl_nhp_data/rr_nl_nhp_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/rr_nl_nhp_data/rr_nl_nhp_test_df.csv", index=False)

"""
SYNTHETIC DATASET

    - The data is generated as follows:
        - generate covariates 'x0', 'x1', 'x2' ~ N(0,1) independently
        - generate survival time from t_star = Weibull(alpha,beta) where alpha, beta are linear combinations of x's
        - randomly select individuals to be censored
        - if uncensored: let t=t_star. 
        - if censored: generate t~Uniform(0,t_star)
    - Train/test split of 50/50.
"""

# set parameters
N=3000 # total number of inds
N_c=400 # number of censored inds
theta_a=[60, 10, -10, 0] # alpha regression parameters
theta_b=[1.2, 0.2, 0, -0.2] # beta regression parameters


# simulate covariates
x = np.random.normal(loc=0.0, scale=1.0, size=(N,3))
df = pd.DataFrame(x, columns=['x{}'.format(i) for i in range(0, 3)]) # make dataframe with colnames x0,x1,x2

# compute Weibull parameters 
alpha = theta_a[0] + x.dot(np.array(theta_a[1:]))
beta = theta_b[0] + x.dot(np.array(theta_b[1:]))
alpha = np.maximum(alpha, 0.01) # ensure >0
beta = np.maximum(beta, 0.01) # ensure >0

# simulate labels
time = alpha * np.random.weibull(beta, size=N) # simulate the death times
status = np.ones(N) # initialise status
censored = random.sample(range(0,N), N_c) # randomly select inds to be censored
status[censored] = 0 # modify status accordingly
time[censored] = np.random.uniform(low=0,high=time[censored]) # sample censoring time using Uniform(0,t).
df["time"] = time 
df["status"] = status

sets = make_train_test(df, 0.5) # make train/test sets

df.to_csv(r"datasets/synthetic_weibull_data/synthetic_weibull_df.csv", index=False)
sets["train_df"].to_csv(r"datasets/synthetic_weibull_data/synthetic_weibull_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/synthetic_weibull_data/synthetic_weibull_test_df.csv", index=False)
