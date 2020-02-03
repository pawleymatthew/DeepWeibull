import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

import torch
import torchtuples as tt

from sklearn_pandas import DataFrameMapper 

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

from data import make_train_test

"""
To implement the DeepHit model, I follow the approach described here:
    https://github.com/havakv/pycox/blob/master/examples/deephit.ipynb
"""


def deep_hit(dataset, alpha=0.3, lr=0.01, epochs=50, batch_size=256):

    """
    Paths to input and output files
    """
    train_path = "datasets/" + dataset + "_data/" + dataset + "_train_df.csv" # training set data
    test_path = "datasets/" + dataset + "_data/" + dataset + "_test_df.csv" # test set data

    training_loss_plot_path = "plots/deep_hit/training_loss/" + dataset + ".png" # create plot of training loss

    """
    Read in the appropriate training and test sets.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    """
    Data preprocessing: set up for DeepHitSingle
    """

    # split the training set into training and validation set
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

    # discretise time for DeepHit, using time index set {0,1,...,T_max}
    num_durations = math.ceil(train_df["time"].max()) # largest survival time in training set set to be T_max 
    labtrans = DeepHitSingle.label_transform(num_durations, scheme='equidistant') # set up partition
    get_target = lambda df: (df['time'].values, df['status'].values) 
    train_y = labtrans.fit_transform(*get_target(train_df))
    val_y = labtrans.transform(*get_target(val_df))
    test_time, test_status = get_target(test_df)

    """
    Define the DeepHit network. Since K=1 (single event) the architecture is very simple.
    Model architecture: layers and widths as stated in paper; use dropout probability 0.1 and batch normalisation 
    """

    p = train_x.shape[1] # number of covariates
    in_features = p # number of input nodes = number of covariates
    out_features = labtrans.out_features # equals num_durations 
    num_nodes = [3*p,5*p,3*p] # layer widths as stated in DeepHit paper
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm=True, dropout=0.2)

    """
    Set learning parameters and fit the model.
    NB: alpha in DeepHitSingle() differs from DH paper: alpha_{pycox} = 1/(1+alpha_{DH}
    The deep_hit(..., alpha, ...) refers to alpha as in the Deep Hit paper.
    """

    alpha_pycox = 1/(1+alpha)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha_pycox, sigma=0.2, duration_index=labtrans.cuts)

    """
    Learning rate: set learning rate automatically or user input
    """

    if lr=="auto":

        lr_finder = model.lr_finder(train_x, train_y, batch_size=256, tolerance=3, lr_range=(1e-7,1e1))
        best_lr = lr_finder.get_best_lr() # find lr with lowest batch loss
        model.optimizer.set_lr(best_lr) # set as model lr

    else: 
        model.optimizer.set_lr(lr)

    """
    Train the model (with early stopping) and plot the training and validation loss.
    """

    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train_x, train_y, batch_size, epochs, callbacks, val_data=(val_x, val_y))

    log.plot()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training loss: DeepHit ($\\alpha =$" + str(alpha) + ") on " + dataset)
    plt.legend(['Train', 'Validation'])
    plt.savefig(training_loss_plot_path)
    plt.clf()
    
    """
    Predict the survival curves for the test set 
    """

    surv = model.predict_surv_df(test_x)

    """
    Create evaluation object
    """

    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

    return ({"test_result" : surv, "ev" : ev})
