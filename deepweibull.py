import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Masking
from keras.optimizers import RMSprop
from keras import backend as k
from sklearn.preprocessing import normalize
from scipy.stats import weibull_min
from scipy.special import gamma

"""
Inputs:
    - outcome_actual = an (N,2) tensor with elements time (time-to-event) and status (0=censored, 1=death).
    - weibull_param_pred = an (N,2) tensor with elements a and b, the predicted Weibull (alpha, beta) parameters. 
Output:
    - the negative log-likelihood, i.e. -logL(weibull_param_pred;outcome_actual)
Description:
    Computes the (negative log-) likelihood of the obersved survival data given Weibull parameters (one pair per individual).
"""

def weibull_loglkhd(outcome_actual, weibull_param_pred, name=None):
    time = outcome_actual[:, 0] # actual time to event
    status = outcome_actual[:, 1] # actual status (censored/dead)
    a = weibull_param_pred[:, 0] # alpha
    b = weibull_param_pred[:, 1] # beta
    norm_time = (time + 1e-35) / a
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
    - tensors: a list of four tensors, as in output of make_train_test()
    - learn_rate: the learning rate of the optimisation procedure
Outputs:
    - model: the final compiled model
    - training_history: a summary of the loss and validation loss at each epoch
    - test_result: a Pandas dataframe with the outcome variables and corresponding predicted Weibull parameters.
"""

def train_deep_weibull(tensors,learn_rate, epochs, steps_per_epoch, validation_steps):

    """
    Define the model:
        - input layer of appropriate length (i.e. number of features)
        - hidden layer of length 32
        - relu activation function
        - hidden layer of length 32
        - relu activation function
        - output layer of length 2
        - exp and softplus activation functions applied (to zeroth and first elements respectively)
    """

    model = Sequential()
    model.add(Dense(3, input_dim=tensors["train_x"].shape[1], activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(2)) # layer with 2 nodes (alpha and beta)
    model.add(Activation(weibull_activate)) # apply custom activation function (exp and softplus)
    """

    Compile the model:
        - using the (negative) log-likelihood for the Weibull as the loss function
        - using Root Mean Square Prop optimisation (common) and customisable learning rate
    """
    model.compile(loss=weibull_loglkhd, optimizer=RMSprop(lr=learn_rate))
    
    """
    Train the model:
        - using the training set (x and y values) 
        - validate on the test set (x and y values)
        - using the user-specified number of epochs, steps_per_epoch, validation_steps
    """
    
    training_history = model.fit(
        tensors["train_x"], 
        tensors["train_y"], 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch, 
        verbose=0,
        validation_data=(tensors["test_x"], tensors["test_y"]), 
        validation_steps=validation_steps)
    
    history = pd.DataFrame(training_history.history)
    history['epoch'] = training_history.epoch
    
    """
    Test the final trained model: 
        - using the training set (x values only)
        - test_result: shows the time/status values alongside the model's predicted Weibull parameters
    """
    
    test_predict = model.predict(tensors["test_x"], steps=1) 
    test_predict = np.resize(test_predict, tensors["test_y"].shape) 
    test_result = np.concatenate((tensors["test_y"], test_predict), axis=1)
    
    return ({
            "model" : model,
            "training_history" : history,
            "test_result" : test_result})

"""
Inputs:
    - tensors: a list of tensors, e.g. as in output of make_train_test(), but only "test_x","test_y" actually get used.
    - model:a trained model, as in "model" output from deep_weibull()
Outputs:
    - a Pandas dataframe with four columns 
        - "time","status" (actual values, from "test_y")
        - "alpha","beta" (the model's predicted Weibull parameters)s
"""

def test_deep_weibull(tensors, model):
    test_predict = model.predict(tensors["test_x"], steps=1) # predict Weibull parameters using covariates
    test_predict = np.resize(test_predict, tensors["test_y"].shape) 
    test_result = np.concatenate((tensors["test_y"], test_predict), axis=1)
    test_result = pd.DataFrame({
        "time" : test_result[:, 0], 
        "status" : test_result[:, 1],
        "alpha" : test_result[:, 2],
        "beta" : test_result[:, 3]
    })
    return test_result