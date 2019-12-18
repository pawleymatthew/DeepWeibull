import pandas as pd
import numpy as np
from make_train_test import make_train_test

"""
Import csv files from DeepHit Github repository
"""

features_url = "https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/METABRIC/cleaned_features_final.csv"
labels_url = "https://raw.githubusercontent.com/chl8856/DeepHit/master/sample%20data/METABRIC/label.csv"

features_df = pd.read_csv(features_url)
labels_df = pd.read_csv(labels_url)

"""
Normalise the features
"""

features_cols = list(features_df.columns)
features_array = np.asarray(features_df)

"""
Input:
    - features : an array containing all the features
Output:
    - features : with all columns normalised (zero mean, unit variance)
"""

def normalise(features):

    for i in range(features.shape[1]):
        if np.std(features[:,i]) != 0:
            features[:,i] = (features[:,i] - np.mean(features[:, i]))/np.std(features[:,i])
        else:
            features[:,i] = (features[:,i] - np.mean(features[:, i]))

    return features

features_df = pd.DataFrame(normalise(features_array), columns=features_cols)

"""
Put into one dataframe and set the label names to time, status
"""

df = pd.concat((features_df, labels_df), axis=1)
df = df.rename({'event_time': 'time', 'label': 'status'}, axis=1)

"""
Simulate the data and write to a csv file.
"""

df.to_csv(r"datasets/metabric_data/metabric_df.csv", index=False)

"""
Split into training and test sets and write these to csv files.
"""

train_frac = 0.9

# make the train/test sets
sets = make_train_test(df, train_frac)

# write to csv files
sets["train_df"].to_csv(r"datasets/metabric_data/metabric_train_df.csv", index=False)
sets["test_df"].to_csv(r"datasets/metabric_data/metabric_test_df.csv", index=False)
