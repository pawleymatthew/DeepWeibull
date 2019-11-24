from synthetic_datasets import synthdata_noise, synthdata_custom, make_train_test
from deepweibull import make_tensors, train_deep_weibull, test_deep_weibull
from simple_models import simple_model_one, simple_model_two

from matplotlib import pyplot as plt
import pandas as pd

"""
Inputs to run simulations:
    - synthdata_type: either "noise" or "custom", depending on what kind of data you want.
    - N: number of individuals, as per input to the synthdata_noise/synthdata_custom functions.
    - N_c: number of censored individuals, as per input to the synthdata_noise/synthdata_custom functions
    - train_frac: proportion of individuals to be allocated to the training set.
    - learn_rate
    - epochs 
    - steps_per_epoch
    - validation_steps
    - init_params_one: initial parameters for optimisation for simple_model_one
    - init_params_two: initial parameters for optimisation for simple_model_two

"""
# Inputs for the synthetic dataset
synthdata_type = "custom"
N = 10
N_c = 1
# Input for making the train/test sets
train_frac = 0.7
# Inputs for the Deep Weibull model
learn_rate = 0.1
epochs = 500
steps_per_epoch = 5
validation_steps = 10
# Initial parameters for simple model one
init_params_one = [50,0,0,0,1]
# Initial parameters for simple model two
init_params_two = [50,0,0,0,1,0,0,0]

"""
Prepare data: 
    - Simulate the dataset
    - Split into train and test sets
"""

# simulate the dataset
synthdata = globals()["synthdata_" + synthdata_type]
df = synthdata(N, N_c)
# create train and test sets
sets = make_train_test(df, train_frac)
train_df = sets["train_df"]
test_df = sets["test_df"]

"""
Run the Deep Weibull model: 
    - Create the tensors
    - Train the model using train_deep_weibull
    - Create plot of the history (training and validation loss on each epoch)
    - Test the model on validation set using test_deep_weibull (create csv file of t,delta,alpha,beta values)
"""

# train the model
tensors = make_tensors(train_df,test_df)
train_deep_weibull = train_deep_weibull(tensors, learn_rate, epochs, steps_per_epoch, validation_steps)
# create plot of the training/validation loss
hist = pd.DataFrame(train_deep_weibull["training_history"].history)
hist['epoch'] = train_deep_weibull["training_history"].epoch
plt.figure()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist['epoch'], hist['loss'], label='Training Loss')
plt.plot(hist['epoch'], hist['val_loss'], label = 'Validation Loss')
plt.ylim([0,5])
plt.legend()
plt.savefig('plt.png', bbox_inches='tight')
# test the model on validation set
test_deep_weibull = test_deep_weibull(tensors, train_deep_weibull["model"])
# create csv file of validation set results
test_deep_weibull.to_csv("deep_weibull_results.csv", index=False)

"""
Run Simple Model One: 
    - Fit the model
    - Use the model to make predictions on validation set 
    - Create csv file of x, t,delta,alpha,beta values)
"""

simple_model_one = simple_model_one(train_df, test_df, init_params_one)
test_result_one = simple_model_one["test_result"]
test_result_one.to_csv("simple_model_one_results.csv", index=False)

"""
Run Simple Model Two: 
    - Fit the model
    - Use the model to make predictions on validation set 
    - Create csv file of x, t, delta, alpha,beta values
"""

simple_model_two = simple_model_two(train_df, test_df, init_params_two)
test_result_two = simple_model_two["test_result"]
test_result_two.to_csv("simple_model_two_results.csv", index=False)


