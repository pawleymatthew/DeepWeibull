import math

def weibull_survival(t, alpha, beta):
    return math.exp(-(t/alpha)**beta)

def weibull_cdf(t, alpha, beta):
    return 1 - weibull_survival(t, alpha, beta)

def weibull_hazard(t, alpha, beta):
    return beta * alpha**(-beta) * t**(beta-1)

def weibull_int_hazard(t, alpha, beta):
    return (t/alpha)**(beta)

def weibull_pdf(t,alpha,beta):
    return weibull_hazard(t, alpha, beta) * weibull_survival(t, alpha, beta)
