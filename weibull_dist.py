import math
import numpy as np

def weibull_surv(t, alpha, beta):

    S = np.empty((len(t),len(alpha)))
    
    for i in range(len(alpha)):
        S[:,i] = np.exp(-np.power(np.divide(t, alpha[i]), beta[i]))

    return S

