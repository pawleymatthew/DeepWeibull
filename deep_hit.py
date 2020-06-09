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
The model architecture and hyperparameters are specified in the paper:
  - learning rate 10e-4
  - batch size = 50
  - early stopping
  - 3 hidden layers of sizes 3p, 5p, 3p respectively
  - dropout probabilty 0.6 on each layer
"""

tidy_datasets = {
  "small_linear_weibull": "Small Linnear Weibull",
  "big_linear_weibull": "Large Linear Weibull",
  "small_nonlinear_weibull": "Small Linear Weibull",
  "big_nonlinear_weibull": "Large Non-Linear Weibull",
  "metabric": "METABRIC",
  "support": "SUPPORT",
  "rrnlnph": "RRNLNPH"
}

def deep_hit(dataset, split, plot=False, alpha=0.5, lr=1e-4, epochs=50, batch_size=100):

    """
    Paths to input and output files
    """
    train_path = "datasets/" + dataset + "/train_" + str(split) + ".csv" # training set data
    test_path = "datasets/" + dataset + "/test_" + str(split) + ".csv" # test set data
 
    training_loss_plot_path = "plots/deep_hit/training_loss/" + dataset + "_" + str(split) + ".pdf" # create plot of loss for different lr's

    """
    Read in the appropriate training and test sets (dataset name and split index)
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    """
    Data preprocessing: set up for DeepHitSingle
    """

    # split the training set into training and validation set
    df = train_df.copy()
    train_df = df.groupby("status").apply(lambda x: x.sample(frac=0.8)) # censoring frac. equal in train and test sets
    train_df = train_df.reset_index(level="status", drop=True)
    train_df = train_df.sort_index()
    val_df = df.drop(train_df.index)

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
    nodes_1 = 3*p
    nodes_2 = 5*p
    nodes_3 = 3*p
    num_nodes = [nodes_1,nodes_2,nodes_3] # layer widths as stated in DeepHit paper
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm=False, dropout=0.25)

    """
    Set learning parameters and fit the model.
    NB: alpha in DeepHitSingle() differs from DH paper: alpha_{pycox} = 1/(1+alpha_{DH}
    The deep_hit(..., alpha, ...) refers to alpha as in the Deep Hit paper.
    """

    alpha_pycox = 1/(1+alpha)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha_pycox, sigma=0.2, duration_index=labtrans.cuts)

    """
    Train the model (with early stopping) and plot the training and validation loss.
    """
    model.optimizer.set_lr(lr)
    callbacks = [tt.callbacks.EarlyStopping()]
    log = model.fit(train_x, train_y, batch_size, epochs, callbacks, val_data=(val_x, val_y))

    if plot ==True: 
      log.plot()
      plt.xlabel("Epoch")
      plt.ylabel("Loss")
      plt.title("Training loss: DeepHit ($\\alpha =$" + str(alpha) + ") on " + tidy_datasets[dataset] + " (Split " + str(split) + ")")
      plt.legend(['Train', 'Validation'])
      plt.savefig(training_loss_plot_path)
      plt.clf()
      plt.close('all')
    
    """
    Predict the survival curves for the test set 
    """

    surv = model.predict_surv_df(test_x)

    """
    Create evaluation object
    """

    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

    return ({"test_result" : surv, "ev" : ev})
