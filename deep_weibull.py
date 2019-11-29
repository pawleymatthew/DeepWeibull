import numpy as np
import random
import math
import pandas as pd
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
    - train_df, test_df: Pandas dataframe as per output of make_train_test()
Outputs:
    - train_x, test_x, train_y, test_y: in tensor form
"""

def make_tensors(train_df, test_df):

    # separate the features and outcome variables
    train_x = train_df.copy()
    train_x = train_x.drop(["true_alpha", "true_beta"], axis=1)
    test_x = test_df.copy()
    test_x = test_x.drop(["true_alpha", "true_beta"], axis=1)
    train_y = pd.DataFrame([train_x.pop(colname) for colname in ['time', 'status']]).T 
    test_y = pd.DataFrame([test_x.pop(colname) for colname in ['time', 'status']]).T
    # convert to tensor
    train_x = tf.convert_to_tensor(train_x.values, tf.float32)
    train_y = tf.convert_to_tensor(train_y.values, tf.float32)
    test_x = tf.convert_to_tensor(test_x.values, tf.float32)
    test_y = tf.convert_to_tensor(test_y.values, tf.float32)

    return ({"train_x" : train_x,
             "train_y" : train_y,
             "test_x" : test_x,
             "test_y" : test_y})

"""
Inputs:
    - tensors: a list of four tensors, as in output of make_tensors()
    - learn_rate: the learning rate of the optimisation procedure
Outputs:
    - model: the final compiled model
    - training_history: a summary of the loss and validation loss at each epoch
    - test_result: a Pandas dataframe with the outcome variables and corresponding predicted Weibull parameters.
"""

def deep_weibull(train_df, test_df, learn_rate=0.01, epochs=100, steps_per_epoch=5, validation_steps=10):

    """
    Make the tensors from the training/test sets
    """
    tensors = make_tensors(train_df, test_df)

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
      
    """
    Test the trained Deep Weibull model on the test set
    """

    test_predict = model.predict(tensors["test_x"], steps=1) # predict Weibull parameters using covariates
    test_predict = np.resize(test_predict, tensors["test_y"].shape) # put into (,2) array
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