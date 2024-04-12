""" Created by : Claire He 
    12.04.24

    Generate minipatches 
functions: 
    - get_minipatch
    - minipatch_regression
    - minipatch_classification **
    - visualise_minipatch
              
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns

def get_minipatch(X_arr,y_arr, x_ratio=0.2, y_ratio=0.2):
    """ Generate a minipatch from a dataset with covariates X, label/response y of size controled by ratio parameters
    Input: 
        X_arr
        y_arr
        x_ratio
        y_ratio
    -------
    Outputs: 
        x_mp
        y_mp
        idx_I
        idx_F """
# get a minipatch of size (n, m)
    N = len(X_arr)
    M = len(X_arr[0])
    n = int(np.round(x_ratio * N))
    m = int(np.round(y_ratio * M))
    r = np.random.RandomState()
    ## index of minipatch
    idx_I = np.sort(r.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp = X_arr[np.ix_(idx_I, idx_F)]
    y_mp = y_arr[np.ix_(idx_I)]
    return x_mp, y_mp, idx_I, idx_F


def minipatch_regression(X_arr, y_arr, model, x_ratio=0.2, y_ratio=0.2, B=10):
    pred = []
    in_mp_obs, in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    for b in range(B):  
        x_mp, y_mp, idx_I, idx_F = get_minipatch(X_arr, y_arr, x_ratio, y_ratio)
        model.fit(x_mp, y_mp)
        pred.append(pd.DataFrame(model.predict(np.array(X_test)[:, idx_F])))
        in_mp_obs[b,idx_I] = True # minipatch b 
        in_mp_feature[b,idx_F] = True
    return [np.array(pred),in_mp_obs,in_mp_feature]

