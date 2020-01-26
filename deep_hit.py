import numpy as np
import pandas as pd
import math
import torch
import torchtuples as tt

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from make_train_test import make_train_test


def deep_hit(train_df, test_df, learn_rate=0.01, epochs=150, batch_size=128, alpha=1, sigma=0.2):

    """
    DESCRIPTION:
        Runs the DeepHit model (with K=1, single event).
    INPUT:
        - train_df : Pandas dataframe
        - test_df : Pandas dataframe
        - num_nodes : list of layer widths (length of list is number of hidden layers)
        - learn_rate: the learning rate of the optimisation procedure
        - epochs :
        - batch_size :
        - alpha : (>0) the hyperparameter in the total loss function (NB: as per DeepHit paper)
        - sigma : (>0) hyperparameter of L_2 loss function
    OUTPUT:
        - model: the final compiled model
        - training_history: a summary of the loss and validation loss at each epoch
        - test_result: a Pandas dataframe with the outcome variables and corresponding predicted Weibull parameters.
    """

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
    Use partition {0,1,...,T_max} where T_max = ceiling of maximum event/censoring time in training set
    """

    num_durations = math.ceil(train_df["time"].max()) # ceiling of maximum event/censoring time in training set
    labtrans = DeepHitSingle.label_transform(num_durations, scheme='equidistant') # set up partition
    get_target = lambda df: (df['time'].values, df['status'].values) 
    train_y = labtrans.fit_transform(*get_target(train_df))
    val_y = labtrans.transform(*get_target(val_df))

    test_time, test_status = get_target(test_df)

    """
    Define the DeepHit model. Since K=1 (single event) the architecture is very simple.
    Model architecture is controlled by function inputs (e.g. layers, batch_norm, dropout, ...) 
    """

    # use MLPVanilla from torchtuples
    p = train_x.shape[1] # number of covariates
    in_features = p # number of input nodes = number of covariates
    out_features = labtrans.out_features # equals num_durations 
    num_nodes = [3*p, 5*p, 3*p] # layer wdiths as stated in DeepHit paper
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm=True, dropout=0.1)

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

    test_result = model.predict_surv_df(test_x)
    test_result = test_result.transpose()
    test_result["time"] = test_time # add the actual event/censoring time to the df
    test_result["status"] = test_status # add the event indicator to the df
    
    return ({
        "model" : model,
        "training_history" : training_history,
        "test_result" : test_result})


def deep_hit_zero_alpha(train_df, test_df, learn_rate=0.01, epochs=150, batch_size=128):

    """
    DESCRIPTION:
        Runs the DeepHit model in deep_hit() (with K=1, single event) with alpha hyperparameter equal to 0.
    """

    # sigma hyperparameter doesn't get used when alpha=0, but must be positive in the function
    epsilon = 1

    return deep_hit(train_df, test_df, learn_rate=learn_rate, epochs=epochs, batch_size=batch_size, alpha=0.0, sigma=epsilon)