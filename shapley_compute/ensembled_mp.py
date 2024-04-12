""" Created by : Claire He 
    12.04.24

    Get shapley ensembled predictor 
              
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from scipy.special import binom 


def mp_shapley(X_arr, res):
    new_shap = np.zeros((X_arr.shape))
    for i in range(X_arr.shape[1]): 
        new_shap[:,i] = shapley_mp(i, res)[:,0]
    return new_shap

def mp_value(feature_subset, pred, in_mp_obs, in_mp_feature, holdout_feature):
    ext_subset = list(feature_subset)
    ext_subset.append(holdout_feature)
    pred, in_mp_obs, in_mp_feature = res 
    pred = np.array(pred)
    selected = np.where(np.any(in_mp_feature[:, ext_subset], axis=1))
    # print('full',np.mean(pred[selected,:], axis = 1)[0].shape)
    return np.mean(pred[selected,:], axis = 1)[0]

def mp_value_holdout(feature_subset, pred, in_mp_obs, in_mp_feature, holdout_feature):
    pred = np.array(pred)
    feat_wo_target = in_mp_feature[~in_mp_feature[:, holdout_feature]]
    selected = np.where(np.any(feat_wo_target[:, feature_subset], axis=1)) # gives the k that correspnod
    # print('holdout',np.mean(pred[selected,:], axis = 1)[0].shape)
    return np.mean(pred[selected,:], axis = 1)[0]

def shapley_mp(target_feature, res):
    pred, in_mp_obs, in_mp_feature = res
    d = len(in_mp_feature[0])
    m = np.sum(in_mp_feature,axis=1)[0]
    n = np.sum(in_mp_obs, axis=1)[0]
    all_features = [list(combinations(set([i for i in range(d)]),j)) for j in range(1,d)] # list of all subsets possible of features
    # all_features[i] contains list of all i-uplet of combinations of features
    features_target = [[combo for combo in all_features[i] if target_feature not in combo] for i in range(len(all_features))] # exclude target feature
    diff = []
    for i in range(m,m+1): # len(features_target)):
        # skip u where |u| < m
        for j in range(len(features_target[i])):
            val_diff = 0 
            feature_subset = features_target[i][j] # u
            # get all patches who's features are contained in u 
            phi_left = mp_value(feature_subset, pred, in_mp_obs, in_mp_feature, holdout_feature=target_feature)
            phi_right = mp_value_holdout(feature_subset, pred,in_mp_obs, in_mp_feature, holdout_feature=target_feature) 
            val_diff += phi_left - phi_right
            diff.append(1/binom(d-1,len(feature_subset)) * val_diff)
    
    shapley_j = sum(diff)

    return shapley_j 
