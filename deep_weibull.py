import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Masking, Dropout
from keras.layers.normalization import BatchNormalization
from keras.callbacks.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras import backend as k

from pycox.evaluation import EvalSurv

from data import make_train_test
from weibull_dist import weibull_surv

"""
CUSTOM LOSS AND ACTIVATION FUNCTIONS
"""

def deep_weibull_loss(y, weibull_param_pred, name=None):
    epsilon = 1e-10
    time = y[:, 0] # actual time to event
    status = y[:, 1] # actual status (censored/dead)
    a = weibull_param_pred[:, 0] # alpha
    b = weibull_param_pred[:, 1] # beta
    norm_time = (time + epsilon) / a # time / alpha (rescaled time)
    return -1 * k.mean(status * (k.log(b) + b * k.log(norm_time)) - k.pow(norm_time, b))


def weibull_activate(weibull_param):
    a = k.exp(weibull_param[:, 0]) # exponential of alpha
    b = k.softplus(weibull_param[:, 1]) # softplus of beta
    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))
    return k.concatenate((a, b), axis=1)


def deep_weibull(dataset, lr=0.01, epochs=50, steps_per_epoch=2):

    """
    Paths to input and output files
    """
    train_path = "datasets/" + dataset + "_data/" + dataset + "_train_df.csv" # training set data
    test_path = "datasets/" + dataset + "_data/" + dataset + "_test_df.csv" # test set data
 
    training_loss_plot_path = "plots/deep_weibull/training_loss/" + dataset + ".png" # create plot of loss for different lr's

    """
    Read in the appropriate training and test sets.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    """
    Data preprocessing: set up for DeepWeibull
    """

    # split the training set into training and validation set
    sets = make_train_test(train_df, train_frac=0.2)
    train_df = sets["train_df"]
    val_df = sets["test_df"]

    # separate covariates and outcomes
    train_x = train_df.copy()
    test_x = test_df.copy()
    val_x = val_df.copy()
    train_y = pd.DataFrame([train_x.pop(colname) for colname in ['time', 'status']]).T
    test_y = pd.DataFrame([test_x.pop(colname) for colname in ['time', 'status']]).T
    val_y = pd.DataFrame([val_x.pop(colname) for colname in ['time', 'status']]).T

    # convert to tensors and float32 type
    train_x = tf.convert_to_tensor(train_x.values, tf.float32)
    train_y = tf.convert_to_tensor(train_y.values, tf.float32)
    test_x = tf.convert_to_tensor(test_x.values, tf.float32)
    test_y = tf.convert_to_tensor(test_y.values, tf.float32)
    val_x = tf.convert_to_tensor(val_x.values, tf.float32)
    val_y = tf.convert_to_tensor(val_y.values, tf.float32)

    """
    Define the DeepWeibull network. 
    Model architecture: layers and widths as stated in paper; use dropout probability 0.1 and batch normalisation 
    """

    p = train_x.shape[1] # number of covariates

    model = Sequential()

    model.add(Dense(2*p, input_dim=p, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(2*p, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(p, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(2)) # layer with 2 nodes (alpha and beta)
    model.add(Activation(weibull_activate)) # apply custom activation function (exp and softplus)

    """
    Compile the model:
        - using the (negative) log-likelihood for the Weibull as the loss function
        - using Root Mean Square Prop optimisation (common) and customisable learning rate
    """

    model.compile(loss=deep_weibull_loss, optimizer=RMSprop(lr=lr))

    """
    Train the model with early stopping and plot the training and validation loss.
    """

    callbacks = [] # use [EarlyStopping()] if desired
    log = model.fit(train_x, train_y, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(val_x, val_y), callbacks=callbacks, validation_steps=5, verbose=1)

    plt.plot(log.history['loss'])
    plt.plot(log.history['val_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Training loss: DeepWeibull on '+ dataset)
    plt.legend(['Train', 'Validation'])
    plt.savefig(training_loss_plot_path)
    plt.clf()

    """
    Use learnt model to make predictions on the test set
    """

    test_predict = model.predict(test_x, steps=1) # predict Weibull parameters using covariates
    test_predict = np.resize(test_predict, test_y.shape) # put into (,2) array
    test_predict = pd.DataFrame(test_predict) # convert to dataframe
    test_predict.columns = ["pred_alpha", "pred_beta"] # name columns
    test_result = test_df.copy()
    test_result.reset_index(inplace = True) # reset the index (before concat - probably better way of doing this)
    test_result = pd.concat([test_result, test_predict], axis=1) # results = test data plus predictions
    test_result.set_index("index", drop=True, inplace=True) # recover the index (after concat - probably better way of doing this)

    """
    Create EvalSurv object
    """
    t_max = train_df["time"].max()
    num_vals = max(math.ceil(t_max), 5000)
    t_vals = np.linspace(0, t_max, num_vals)
    surv = weibull_surv(t_vals, test_result["pred_alpha"], test_result["pred_beta"])
    surv = pd.DataFrame(data=surv, index=t_vals)

    test_time = test_df['time'].values
    test_status = test_df['status'].values

    ev = EvalSurv(surv, test_time, test_status, censor_surv='km')

    return ({"test_result" : test_result, "ev" : ev})

go = deep_weibull("support")