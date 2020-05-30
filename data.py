import pandas as pd
import numpy as np
import random
import math
from pycox.datasets import metabric, support, rr_nl_nhp

"""
PREPROCESSING FUNCTIONS
"""

def normalise(df, colnames):
    df[colnames] = df[colnames].apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    return df
    
def make_train_test(df, train_frac, dataset, n_splits=3):

    full_file_path = "datasets/" + dataset + "/full.csv"
    df.to_csv(full_file_path, index=False)
    
    for i in range(1, n_splits+1):

        # set seed equal to loop index
        random.seed(123*i)

        # set output file paths
        train_file_path = "datasets/" + dataset + "/train_" + str(i) + ".csv"
        test_file_path = "datasets/" + dataset + "/test_" + str(i) + ".csv"

        # create splits (different each time - sample depends on seed)
        train_df = df.groupby("status").apply(lambda x: x.sample(frac=train_frac)) # censoring frac. equal in train and test sets
        train_df = train_df.reset_index(level="status", drop=True)
        train_df = train_df.sort_index()
        test_df = df.drop(train_df.index)

        # save the resulting file
        train_df.to_csv(train_file_path, index=False)
        test_df.to_csv(test_file_path, index=False)


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
make_train_test(df, train_frac=0.8, dataset="metabric", n_splits=3) # make train/test splits

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
make_train_test(df, train_frac=0.8, dataset="support", n_splits=3) # make train/test splits

"""
RRNLNPH
    - No columns need to be normalised.
    - Columns 'duration_true', 'event_true', 'censoring_true' contain extraneous intermediate information about the simulation.
    - Change ['duration', 'event'] -> ['time', 'status'].
    - Train/test split of 60/40.
"""

df = rr_nl_nhp.read_df() # read in dataset (note name sic)
df = df.drop(['duration_true', 'event_true', 'censoring_true'], axis=1) # remove extraneous cols
df.rename(columns={"duration": "time", "event": "status"}, inplace=True) # rename duration/event cols
make_train_test(df, train_frac=0.6, dataset="rrnlnph", n_splits=3) # make train/test splits

"""
LINEAR WEIBULL:

    - The data is generated as follows:
        - generate covariates 'x0', 'x1', ... 'x4' ~ N(0,1) independently
        - generate survival time from t_star = Weibull(alpha,beta) where alpha, beta are linear combinations of x's
        - randomly select individuals to be censored
        - if uncensored: let t=t_star. 
        - if censored: generate t~Uniform(0,t_star)
    - Censoring fraction is 20%.
    - SMALL : Training = 300, Test = 39700
    - BIG : Training = 30000, Test = 10000
"""

random.seed(1234)

# set parameters
N=40000 # total number of inds
N_c=math.ceil(0.2*N) # number of censored inds
theta=[50, 25, -25, 0, 0, 0] # alpha regression parameters

# simulate covariates
x = np.random.uniform(low=-1.0, high=1.0, size=(N,5))
df = pd.DataFrame(x, columns=['x{}'.format(i) for i in range(0, 5)]) # make dataframe with colnames x0,x1,x2,x3,x4

# compute Weibull parameters 
alpha = theta[0] + x.dot(np.array(theta[1:]))
beta = [1.0]
alpha = np.maximum(alpha, 0.01) # ensure alpha>0 (very low prob it is <0).

# simulate labels
time = alpha * np.random.weibull(beta, size=N) # simulate the death times
status = np.ones(N) # initialise status
censored = random.sample(range(0,N), N_c) # randomly select inds to be censored
status[censored] = 0 # modify status accordingly
time[censored] = np.random.uniform(low=0,high=time[censored]) # sample censoring time using Uniform(0,t).
df["time"] = time 
df["status"] = status
df.loc[df['time'] > 500, 'status'] = 1
df.loc[df['time'] > 500, 'time'] = 500

# create BIG datasets (train and test splits)
make_train_test(df, train_frac=30000/N, dataset="big_linear_weibull", n_splits=3) # make train/test splits

# create SMALL datasets (train and test splits)
#df.groupby("status").apply(lambda y: y.sample(frac=10300/N)) # censoring frac. equal in small and big sets
make_train_test(df, train_frac=300/N, dataset="small_linear_weibull", n_splits=3) # make train/test splits


"""
NON-LINEAR WEIBULL:

    - The data is generated as follows:
        - generate covariates 'x0', 'x1', ... 'x4' ~ N(0,1) independently
        - generate survival time from t_star = Weibull(alpha,beta) where alpha, beta are linear combinations of x's
        - randomly select individuals to be censored
        - if uncensored: let t=t_star. 
        - if censored: generate t~Uniform(0,t_star)
    - Censoring fraction is 20%.
    - SMALL : Training = 300, Test = 39700
    - BIG : Training = 30000, Test = 10000
"""

random.seed(1234)

# set parameters
N=40000 # total number of inds
N_c=math.ceil(0.2*N) # number of censored inds
fun = lambda x :  80 - 40*x[0]**2 + 30*x[0]*x[1]  # alpha as function of covaraties,

# simulate covariates
x = np.random.uniform(low=-1.0, high=1.0, size=(N,5))
df = pd.DataFrame(x, columns=['x{}'.format(i) for i in range(0, 5)]) # make dataframe with colnames x0,x1,x2,x3,x4

# compute Weibull parameters 
alpha = fun(np.transpose(x))
beta = [1.1]
alpha = np.maximum(alpha, 0.01) # ensure alpha>0 (very low prob it is <0).

# simulate labels
time = alpha * np.random.weibull(beta, size=N) # simulate the death times
status = np.ones(N) # initialise status
censored = random.sample(range(0,N), N_c) # randomly select inds to be censored
status[censored] = 0 # modify status accordingly
time[censored] = np.random.uniform(low=0,high=time[censored]) # sample censoring time using Uniform(0,t).
df["time"] = time 
df["status"] = status
df.loc[df['time'] > 500, 'status'] = 1
df.loc[df['time'] > 500, 'time'] = 500

# create BIG datasets (train and test splits)
make_train_test(df, train_frac=30000/N, dataset="big_nonlinear_weibull", n_splits=3) # make train/test splits

# create SMALL datasets (train and test splits)
#df.groupby("status").apply(lambda x: x.sample(frac=10300/40000)) # censoring frac. equal in small and big sets
make_train_test(df, train_frac=300/N, dataset="small_nonlinear_weibull", n_splits=3) # make train/test splits