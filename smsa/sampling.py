import pandas as pd
import numpy as np
import random

# source: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/sandbox/distributions/multivariate.py
# written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal


def generate_logrets_mtx(blending_param, sample_size, mean, cov, columns):
    raw_data = []
    random.seed()

    for i in range(0, sample_size):
        # blending dice roll
        blending_value = random.uniform(0, 1)
        blending_flag = blending_value <= blending_param
        
        # generate multivariate normal or student random vector
        if blending_flag:
            raw_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=1)[0])
        else:
            raw_data.append(multivariate_t_rvs(mean, cov, df=3, n=1)[0])

    logrets_mtx = pd.DataFrame(columns=columns, data=raw_data)
    return logrets_mtx  