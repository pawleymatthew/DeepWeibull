import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.models import DeepHitSingle
from make_train_test import make_train_test

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper 

import matplotlib.pyplot as plt

"""
Inputs:
    - train_df : Pandas dataframe
    - test_df : Pandas dataframe
    - num_durations : number of timepoints to use in discretisation
    - discrete_scheme : scheme used to discretise time, either 'quantiles' (default) or 'equidistant'
    - num_nodes : list of layer widths (length of list is number of hidden layers)
    - learn_rate: the learning rate of the optimisation procedure
    - epochs :
    - batch_size :
    - batch_norm : 
    - dropout : 
    - alpha : (>0) the hyperparameter in the total loss function (NB: as per DeepHit paper)
    - sigma : (>0) hyperparameter of L_2 loss function
    - smooth_points : number of interpolation points used for smoothing the survival curve (0 = no smoothing)
Outputs:
    - model: the final compiled model
    - training_history: a summary of the loss and validation loss at each epoch
    - test_result: a Pandas dataframe with the outcome variables and corresponding predicted Weibull parameters.
"""

def deep_hit(train_df, test_df, num_durations=19, discrete_scheme='quantiles', num_nodes=[32,32], learn_rate=0.02, epochs=150, batch_norm=True, batch_size=32, dropout=0.1, alpha=0.1, sigma=0.1, smooth_points=10):

    """
    Make validation set, separate covariates and outcomes, then convert the covariates to float32
    """

    sets = make_train_test(train_df, train_frac=0.2)
    train_df = sets["train_df"]
    val_df = sets["test_df"]

    # convert x values to float32 (needed for PyTorch)
    x_cols = list(train_df)
    x_cols.remove('time')
    x_cols.remove('status')
    x_cols = [(col, None) for col in x_cols]
    x_mapper = DataFrameMapper(x_cols)
    train_x = x_mapper.fit_transform(train_df).astype('float32')
    val_x = x_mapper.transform(val_df).astype('float32')
    test_x = x_mapper.transform(test_df).astype('float32')

    """
    DeepHit is a discrete-time model, so need to discretise the time points. 
    Can either discretise with equidistant grid or gridpoints can be determined by quantiles.
    Function input discrete_scheme controls this (default: 'quantiles').
    The grid has num_durations time points.
    """

    labtrans = DeepHitSingle.label_transform(num_durations, scheme=discrete_scheme)
    get_target = lambda df: (df['time'].values, df['status'].values)
    train_y = labtrans.fit_transform(*get_target(train_df))
    val_y = labtrans.transform(*get_target(val_df))

    """
    Define the DeepHit model. Since K=1 (single event) the architecture is very simple.
    Model architecture is controlled by function inputs (e.g. layers, batch_norm, dropout, ...) 
    """

    # use MLPVanilla from torchtuples
    in_features = train_x.shape[1] # number of input nodes = number of covariates
    out_features = labtrans.out_features # equals num_durations 
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

    """
    Set learning parameters and fit the model.
    NB: alpha in DeepHitSingle() differs from DH paper: here 1=log-likelihood only, 0=concordance only
    The alpha in my deep_hit() refers to alpha in the DeepHit paper (alpha >=0)
    I tranform it here: alpha_{pycox} = 1/(1+alpha_{DH} < 1
    """
    alpha_pycox = 1/(1+alpha)

    # define the model using previously defined net
    model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha_pycox, sigma=sigma, duration_index=labtrans.cuts)

    model.optimizer.set_lr(learn_rate) # set learning rate
    callbacks = [tt.callbacks.EarlyStopping()] # use early stopping

    # fit the model using the training set (with validation set)
    training_history = model.fit(train_x, train_y, batch_size, epochs, callbacks, val_data=(val_x, val_y), verbose=0)
    
    """
    Predict the survival curves for the test set individuals
    """

    test_result = model.interpolate(smooth_points).predict_surv_df(test_x)

    return ({
        "model" : model,
        "training_history" : training_history,
        "test_result" : test_result})

"""
Testing it works...
"""

dataset_name = "metabric"

# filepaths of input csv files
train_path = "datasets/" + dataset_name + "_data/" + dataset_name + "_train_df.csv"
test_path = "datasets/" + dataset_name + "_data/" + dataset_name + "_test_df.csv"

# filepath of output csv file
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

run = deep_hit(train_df, test_df)
surv = run["test_result"]
print(surv)

surv.iloc[:, :50].plot(drawstyle='steps-post')
plt.ylabel('S(t)')
_ = plt.xlabel('t')
plt.savefig('deep_hit_S_curve.png')