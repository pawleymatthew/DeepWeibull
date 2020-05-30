import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tidy_models = {
  "regression_weibull": "Regression Weibull",
  "deep_weibull": "Deep Weibull",
  "deep_hit": "Deep Hit",
  "deep_hit_zero_alpha": "Deep Hit ($\\alpha =0$)"
}

tidy_datasets = {
  "small_linear_weibull": "Small Linear Weibull",
  "big_linear_weibull": "Large Linear Weibull",
  "small_nonlinear_weibull": "Small Linear Weibull",
  "big_nonlinear_weibull": "Large Non-Linear Weibull",
  "metabric": "METABRIC",
  "support": "SUPPORT",
  "rrnlnph": "RRNLNPH"
}


def get_survival_curves(dataset, split, i):

    # weibull survival function
    S = lambda t,a,b: np.exp(-1*np.power(np.divide(t,a),b)) # survival function of Weibull(a,b)

    """
    Get the predicted alpha/beta values from RegWeib
    """
    rw_test_path = "test_results/regression_weibull/" + dataset + "_" + str(split) + ".pkl"
    rw_test_df = pd.read_pickle(rw_test_path)
    rw_pred_alpha = rw_test_df.loc[i,"pred_alpha"]
    rw_pred_beta = rw_test_df.loc[i,"pred_beta"]
   
    """
    Get the predicted alpha/beta values from DeepWeib
    """
    dw_test_path = "test_results/deep_weibull/" + dataset + "_" + str(split) + ".pkl"
    dw_test_df = pd.read_pickle(dw_test_path)
    dw_pred_alpha = dw_test_df.loc[i,"pred_alpha"]
    dw_pred_beta = dw_test_df.loc[i,"pred_beta"]

    """ 
    Get the predicted S(t) values for DeepHit
    """
    dh_test_path = "test_results/deep_hit/" + dataset + "_" + str(split) + ".pkl"
    dh_test_df = pd.read_pickle(dh_test_path)
    S_dh = dh_test_df.iloc[:,i]
    dh_t_vals = dh_test_df.index.values.tolist() # list of the t values

    # compute the survival curves at t_vals

    if dataset=="metabric":
        weibull_t_vals = np.arange(start=0.001,stop=351.0,step=0.2) 

    if dataset=="support":
        weibull_t_vals = np.arange(start=0.001,stop=2029.0,step=0.2)

    S_rw = S(weibull_t_vals, rw_pred_alpha, rw_pred_beta)
    S_dw = S(weibull_t_vals, dw_pred_alpha, dw_pred_beta)

    return ({
        "weibull_t_vals" : weibull_t_vals, 
        "dh_t_vals" : dh_t_vals,
        "S_rw" : S_rw,
        "S_dw" : S_dw,
        "S_dh" : S_dh
        })

def plot_survival_curves_real(dataset="metabric", split=1):
    
    # path to output file
    plot_file_path = "plots/real_data_experiments/survival_curves/" + dataset + "_" + str(split) + ".pdf" 

    patient_33 = get_survival_curves(dataset, split, i=33)
    patient_84 = get_survival_curves(dataset, split, i=84)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,12)) # 3X1 subplots

    l11 = ax1.plot(patient_33["weibull_t_vals"], patient_33["S_rw"], color="blue")[0]
    l12 = ax1.plot(patient_33["weibull_t_vals"], patient_33["S_dw"], color="orange")[0]
    l13 = ax1.plot(patient_33["dh_t_vals"], patient_33["S_dh"], color="green")[0]

    ax2.plot(patient_84["weibull_t_vals"], patient_84["S_rw"], color="blue")[0]
    ax2.plot(patient_84["weibull_t_vals"], patient_84["S_dw"], color="orange")[0]
    ax2.plot(patient_84["dh_t_vals"], patient_84["S_dh"], color="green")[0]

    ax1.set_xlabel(r'$t$')
    ax2.set_xlabel(r'$t$')

    ax1.set_ylabel(r'$\hat{S}(t)$')
    ax2.set_ylabel(r'$\hat{S}(t)$')

    ax1.set_title("Patient 33 (" + tidy_datasets[dataset]+ ": Split "+ str(split)+")")
    ax2.set_title("Patient 84 (" + tidy_datasets[dataset]+ ": Split "+ str(split)+")")

    # create legend
    fig.legend([l11, l12, l13],     # The line objects
           labels=["RegressionWeibull","DeepWeibull","DeepHit"],   # The labels for each line
           loc="lower center",
           )

    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.12)
    plt.savefig(plot_file_path)

plot_survival_curves_real()

"""
Plot Brier scores for each split in one figure
"""

for dataset in ["metabric", "support"]:

    # path for saving Brier score plots
    brier_score_plot_path = "plots/brier_scores/" + dataset + "/" + dataset + "_singlefig.pdf"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,12)) # 3X1 subplots

    # split 1 data
    df_1_path = "evaluation_metrics_results/brier_scores/" + dataset + "_1.pkl"
    df_1 = pd.read_pickle(df_1_path)
    x1 = df_1["t"]
    y11 = df_1["regression_weibull"]
    y12 = df_1["deep_weibull"]
    y13 = df_1["deep_hit"]
    y14 = df_1["deep_hit_zero_alpha"]

    # split 2 data
    df_2_path = "evaluation_metrics_results/brier_scores/" + dataset + "_2.pkl"
    df_2 = pd.read_pickle(df_2_path)
    x2 = df_2["t"]
    y21 = df_2["regression_weibull"]
    y22 = df_2["deep_weibull"]
    y23 = df_2["deep_hit"]
    y24 = df_2["deep_hit_zero_alpha"]

    # split 3 data
    df_3_path = "evaluation_metrics_results/brier_scores/" + dataset + "_3.pkl"
    df_3 = pd.read_pickle(df_3_path)
    x3 = df_3["t"]
    y31 = df_3["regression_weibull"]
    y32 = df_3["deep_weibull"]
    y33 = df_3["deep_hit"]
    y34 = df_3["deep_hit_zero_alpha"]

    # create subplots

    l11 = ax1.plot(x1, y11, color="blue")[0]
    l12 = ax1.plot(x1, y12, color="orange")[0]
    l13 = ax1.plot(x1, y13, color="green")[0]
    l14 = ax1.plot(x1, y14, color="red")[0]

    l21 = ax2.plot(x2, y21, color="blue")[0]
    l22 = ax2.plot(x2, y22, color="orange")[0]
    l23 = ax2.plot(x2, y23, color="green")[0]
    l24 = ax2.plot(x2, y24, color="red")[0]

    l31 = ax3.plot(x3, y31, color="blue")[0]
    l32 = ax3.plot(x3, y32, color="orange")[0]
    l33 = ax3.plot(x3, y33, color="green")[0]
    l34 = ax3.plot(x3, y34, color="red")[0]
         

    ax1.set_xlabel("Time")
    ax2.set_xlabel("Time")
    ax3.set_xlabel("Time")

    ax1.set_ylabel("Brier Score")
    ax2.set_ylabel("Brier Score")
    ax3.set_ylabel("Brier Score")

    ax1.set_title(tidy_datasets[dataset]+": Split 1")
    ax2.set_title(tidy_datasets[dataset]+": Split 2")
    ax3.set_title(tidy_datasets[dataset]+": Split 3")

    #fig.suptitle(tidy_datasets[dataset])

    # create legend
    fig.legend([l11, l12, l13, l14],     # The line objects
           labels=tidy_models.values(),   # The labels for each line
           loc="lower center",
           )

    fig.tight_layout()
    plt.subplots_adjust(bottom = 0.12)
    plt.savefig(brier_score_plot_path)
