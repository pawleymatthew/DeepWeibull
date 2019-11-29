import pandas as pd

"""
Description:
    - Splits a dataset into training and test sets, each with equal censoring fractions. 
Inputs:
    - df :  a pd dataframe (must have column called "status" indicating censoring status)
    - train_frac : number in [0,1], proportion of observations to allocate to training set
Outputs:
     - train_df :  a pd dataframe with the training observations
     - test_df: a pd dataframe with the test observations
"""

def make_train_test(df, train_frac):
    # randomly select inds to go in train/test sets
    train_df = df.groupby("status").apply(lambda x: x.sample(frac=train_frac))
    train_df = train_df.reset_index(level="status", drop=True)
    train_df = train_df.sort_index()
    test_df = df.drop(train_df.index)
    # return named list of dataframes
    return ({"train_df" : train_df,
             "test_df" : test_df})