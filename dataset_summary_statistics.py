import pandas as pd
import numpy as np

def summary_statistics(dataset):

    # read in dataframe
    data_path = "datasets/" + dataset + "/full.csv"
    df = pd.read_csv(data_path)

    N = df.shape[0]
    p = df.shape[1] -2


    cens_df = df[df.status == 0]
    N_c = cens_df.shape[0]
    cens_frac = N_c/N

    c_min = cens_df["time"].min()
    c_med = cens_df["time"].median()
    c_max = cens_df["time"].max()

    event_df = df[df.status == 1]
    t_min = event_df["time"].min()
    t_med = event_df["time"].median()
    t_max = event_df["time"].max()

    print ("Number of obs.: " + str(N))
    print ("Number of features.: " + str(p))
    print ("Censoring fraction: " + str(cens_frac))
    print ("Min event time: " + str(t_min))
    print ("Median event time: " + str(t_med))
    print ("Max event time: " + str(t_max))
    print ("Min censoring time: " + str(c_min))
    print ("Median censoring time: " + str(c_med))
    print ("Max censoring time: " + str(c_max))


for dataset in ["metabric", "support"]:
    summary_statistics(dataset)