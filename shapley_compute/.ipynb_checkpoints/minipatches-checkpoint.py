""" Created by : Claire He 
    12.04.24

    Generate minipatches 
functions: 
    - get_minipatch
    - minipatch_regression, minipatch_regression_loo
    - minipatch_classification **
    - visualise_minipatch
              
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns

palette = sns.color_palette([
    "#7fbf7b",  # Light Green
    "#af8dc3",  # Lavender
    "#e7d4e8",  # Light Purple
    "#fdc086",  # Light Orange
    "#ff9896",  # Light Red
    "#c5b0d5"   # Light Blue
])
def get_minipatch(X_arr,y_arr, x_ratio, seed=None):
    """ Generate a minipatch from a dataset with covariates X, with obs size controled by ratio parameters
    Input: 
        X_arr
        y_arr
        x_ratio
    -------
    Outputs: 
        x_mp
        y_mp
        idx_I
        idx_F """
    N = X_arr.shape[0]
    M = X_arr.shape[1]
    
    # get a random feature size
    m =  np.random.choice([i for i in range(1,M)])
    assert int(np.round(x_ratio * N)) > m # verify that enough observations are sampled
    n = int(np.round(x_ratio * N))
    if seed==None:
        r = np.random.RandomState()
    else:
        r = np.random.RandomState(seed)
    ## index of minipatch
    # print(n, N)
    idx_I = np.sort(r.choice(N, size=n, replace=False)) # uniform sampling of subset of observations
    idx_F = np.sort(r.choice(M, size=m, replace=False)) # uniform sampling of subset of features
    ## record which obs/features are subsampled 
    x_mp = X_arr[np.ix_(idx_I, idx_F)]
    y_mp = y_arr[np.ix_(idx_I)]
    return x_mp, y_mp, idx_I, idx_F

def minipatch_regression(X_arr, y_arr, Xi, model, x_ratio, B=1000, plot_prop=False):
    """ Fit the minipatch ensemble estimator on the training data and predict on test set Xi
    Input:
        X_arr: training predictors
        y_arr: training set response
        Xi: test set
        model: chosen model for regression
        x_ratio: ratio of observation to sample from
        B: number of replicates
        plot_prop: if True, plots the minipatch feature coverage histogram
    -------
    Outputs: 
        [np.array, np.array, np.array]: prediction on test set, boolean dictionary of minipatch observations, boolean dictionary of minipatch features
    """
    pred = []
    mp_feat_size = []
    N = X_arr.shape[0]
    M = X_arr.shape[1]
    in_mp_obs, in_mp_feature = np.zeros((B,N),dtype=bool),np.zeros((B,M),dtype=bool)
    for b in range(B):  
        x_mp, y_mp, idx_I, idx_F = get_minipatch(X_arr, y_arr, x_ratio)
        mp_feat_size.append(len(idx_F))
        model.fit(x_mp, y_mp)
        pred.append(pd.DataFrame(model.predict(np.array(Xi)[:, idx_F])))
        in_mp_obs[b,idx_I] = True # minipatch b 
        in_mp_feature[b,idx_F] = True
    if plot_prop:
        plt.hist(mp_feat_size)
        plt.suptitle('Minipatch length histogram')
    
    return [np.array(pred),in_mp_obs,in_mp_feature]

def minipatch_regression_loo(X_arr, y_arr, model, x_ratio, B=1000, r=None, plot_prop=False):
    """ Fit the minipatch ensemble estimator on the training data and predict on leave-one-out Xi
    Input:
        X_arr: training predictors
        y_arr: training set response
        Xi: test set
        model: chosen model for regression
        x_ratio: ratio of observation to sample from
        B: number of replicates
        plot_prop: if True, plots the minipatch feature coverage histogram
    -------
    Outputs: 
        [np.array, np.array, np.array]: prediction on test set, boolean dictionary of minipatch observations, boolean dictionary of minipatch features
    """
    pred = []
    mp_feat_size = []
    N = X_arr.shape[0]
    M = X_arr.shape[1]
    in_mp_obs, in_mp_loo, in_mp_feature = np.zeros((B,N),dtype=bool), np.zeros((B,N),dtype=bool), np.zeros((B,M),dtype=bool)
    for b in range(B):  
        x_mp, y_mp, idx_I, idx_F = get_minipatch(X_arr, y_arr, x_ratio, r)
        mp_feat_size.append(len(idx_F))
        model.fit(x_mp[:-1,:], y_mp[:-1]) # leave-one-out
        Xi = x_mp[-1,:]
        pred.append(pd.DataFrame(model.predict(Xi.reshape(1,-1))))
        in_mp_obs[b,idx_I[:-1]] = True # minipatch b train points
        in_mp_feature[b,idx_F] = True
        in_mp_loo[b, idx_I[-1]] = True # test point loo
    if plot_prop:
        plt.hist(mp_feat_size)
        plt.suptitle('Minipatch length histogram')
    
    return [np.array(pred),in_mp_obs,in_mp_loo, in_mp_feature]

def visualise_minipatch(in_mp_obs, in_mp_feature, color_palette = palette, type='sorted'):
    
    B = in_mp_obs.shape[0]
    matrix = np.zeros((in_mp_obs.shape[1],in_mp_feature.shape[1]))
    for i in range(B):
        matrix += (in_mp_obs[i][:, np.newaxis] & in_mp_feature[i]).astype(int)
    df = pd.DataFrame(matrix, columns = X.columns)
    if type =='sorted':
        sns.heatmap(df[df.mean().sort_values().index].sort_values(by=df[df.mean().sort_values().index].columns[-1], axis=0), cmap=palette)
    else:
        sns.heatmap(df, cmap=palette)
    plt.title('Patch selection frequency')