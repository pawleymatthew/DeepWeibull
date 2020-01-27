import pandas as pd

from deep_weibull import deep_weibull
from deep_hit import deep_hit, deep_hit_zero_alpha
from simple_models import simple_model_one, simple_model_two

def run_model(model_name, dataset_name):

    """
    Filepaths of the input and output files
    """

    # filepaths of input csv files
    train_path = "datasets/" + dataset_name + "_data/" + dataset_name + "_train_df.csv"
    test_path = "datasets/" + dataset_name + "_data/" + dataset_name + "_test_df.csv"
    # filepath of output csv file
    results_path = "predict_results/" + model_name + "_~_" + dataset_name + "_results.csv"

    """
    Read in input files, the train and test sets.
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    """
    Run the model on the dataset and write the test_result dataframe (i.e. test_df with the predicted alpha, beta values) to a csv
    """

    # run the model on the dataset
    model = globals()[model_name]
    run = model(train_df, test_df)
    test_result_df = run["test_result"]
    # write to csv
    test_result_df.to_csv(results_path, index=False)
    
"""
Repeat for all models and datasets
"""

models = ["simple_model_one", "simple_model_two", "deep_weibull", "deep_hit", "deep_hit_zero_alpha"]
datasets = ["linear_weibull", "non_linear_weibull","metabric"]

models = ["deep_weibull"]
datasets = ["linear_weibull", "non_linear_weibull"]

for model_name in models:
    for dataset_name in datasets:
        run_model(model_name, dataset_name)

"""
Before running compare_performance.py, check that DeepWeibull has worked on METABRIC.
"""