import numpy as np
import random
import math
import pandas as pd
import tensorflow as tf
from make_train_test import make_train_test
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Masking, Dropout
from keras.optimizers import RMSprop
from keras import backend as k
from sklearn.preprocessing import normalize
from scipy.stats import weibull_min
from scipy.special import gamma

"""
Inputs:
    - y = an (N,2) tensor with elements time (time-to-event) and status (0=censored, 1=death).
    - weibull_param_pred = an (N,2) tensor with elements a and b, the predicted Weibull (alpha, beta) parameters. 
Output:
    - the total loss (currently only coded to have loglkhd term, no concordance term)
Description:
    Computes the (negative log-) likelihood of the obersved survival data given Weibull parameters (one pair per individual).
"""

def deep_weibull_loss(y, weibull_param_pred, name=None):
    epsilon = 1e-35
    time = y[:, 0] # actual time to event
    status = y[:, 1] # actual status (censored/dead)
    a = weibull_param_pred[:, 0] # alpha
    b = weibull_param_pred[:, 1] # beta
    norm_time = (time + epsilon) / a # time / alpha (rescaled time)
    return -1 * k.mean(status * (k.log(b) + b * k.log(norm_time)) - k.pow(norm_time, b))

"""
N = number of observations (in training set).
Inputs:
    - weibull_param = an (N,2) tensor with elements a and b, the Weibull parameters before the activation layer
Output:
    - (a,b) = the Weibull paramters after the activation layer
Description:
    A custom activation function that applies expontial to alpha and softplus to beta in the final layer.
"""

def weibull_activate(weibull_param):
    a = k.exp(weibull_param[:, 0]) # exponential of alpha
    b = k.softplus(weibull_param[:, 1]) # softplus of beta
    a = k.reshape(a, (k.shape(a)[0], 1))
    b = k.reshape(b, (k.shape(b)[0], 1))
    return k.concatenate((a, b), axis=1)

"""
Inputs:
    - train_df: Pandas dataframe
    - test_df : Pandas dataframe
    - learn_rate: the learning rate of the optimisation procedure
    - epochs : number of epochs for learning
    - steps per epoch :
    - validation steps :
    - dropout : dropout probability applied to each layer
Outputs:
    - model: the final compiled model
    - training_history: a summary of the loss and validation loss at each epoch
    - test_result: a Pandas dataframe with the outcome variables and corresponding predicted Weibull parameters.
"""

def deep_weibull(train_df, test_df, learn_rate=0.01, epochs=150, steps_per_epoch=5, validation_steps=10, dropout=0.1):

    """
    Make validation set, separate covariates and outcomes, then convert the data to tensors
    """

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
    Define the model:
        - input layer of appropriate length (i.e. number of features)
        - hidden layer (number covariates)
        - relu activation function
        - hidden layer (2 * number covariates)
        - relu activation function
        - output layer (number covariates)
        - exp and softplus activation functions applied (to zeroth and first elements respectively)
        - dropout probability applied to each layer
    """

    p = train_x.shape[1] # number of covariates

    model = Sequential()
    model.add(Dense(2*p, input_dim=p, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(2*p, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(p, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(2)) # layer with 2 nodes (alpha and beta)
    model.add(Activation(weibull_activate)) # apply custom activation function (exp and softplus)
    
    """
    Compile the model:
        - using the (negative) log-likelihood for the Weibull as the loss function
        - using Root Mean Square Prop optimisation (common) and customisable learning rate
    """

    model.compile(loss=deep_weibull_loss, optimizer=RMSprop(lr=learn_rate))
    
    """
    Train the model:
        - using the training set (x and y values) 
        - validate on the test set (x and y values)
        - using the user-specified number of epochs, steps_per_epoch, validation_steps
    """

    training_history = model.fit(
        train_x, 
        train_y, 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        verbose=0,
        validation_data=(val_x, val_y), 
        validation_steps=validation_steps)
      
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

    return ({
            "model" : model,
            "training_history" : training_history,
            "test_result" : test_result})