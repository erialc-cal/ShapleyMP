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
from itertools import combinations

def mp_shapley(Xi, res):
    new_shap = np.zeros((Xi.shape))
    for i in range(Xi.shape[1]): 
        new_shap[:,i] = shapley_mp(i, res)
    return new_shap


def shapley_mp(target_feature, res):
    pred, in_mp_obs, in_mp_feature = res
    d = len(in_mp_feature[0])
    m = np.sum(in_mp_feature,axis=1)[0]
    n = np.sum(in_mp_obs, axis=1)[0]
    all_features = [list(combinations(set([i for i in range(d)]),j)) for j in range(1,d)] # list of all subsets possible of features
    # all_features[i] contains list of all i-uplet of combinations of features
    features_target = [[combo for combo in all_features[i] if target_feature not in combo] for i in range(len(all_features))] # exclude target feature
    diff = []
    for i in range(len(features_target)):
        for j in range(len(features_target[i])):
        
            val_diff = np.zeros(pred.shape[1])
            feature_subset = features_target[i][j]
            target_indices = np.where(in_mp_feature[:,target_feature])[0] # Get the indices of the rows where the target column is True
            
            oh = np.zeros(d, dtype=bool)
            oh[[feature_subset]] = True # one hot feature subset
            oh_j = np.zeros(d, dtype=bool)
            l = list(feature_subset)
            l.append(target_feature)
            oh_j[[l]] = True # one hot feature subset + target
            
            mask = np.all(in_mp_feature == oh, axis=1) # mask the feature subset 
            right_indices = np.where(mask)[0] # find matching indices of mask 
            mask_j = np.all(in_mp_feature == oh_j, axis=1) # mask the feature subset + target
            left_indices = np.where(mask_j)[0] 
            
            if (len(left_indices) ==0) or (len(right_indices)==0) :
                pass
            else:
                augmented_pred = pred[list(left_indices),:,0]
                # augmented_idx = in_mp_obs[list(left_indices),:]
                selected_pred = pred[list(right_indices),:,0]
                # selected_idx = in_mp_obs[list(right_indices),:]
                phi_right = np.mean(selected_pred) # np.mean(np.array([selected_pred[i,:][selected_idx[i,:]] for i in range(selected_idx.shape[0])]),axis=0)
                phi_left = np.mean(augmented_pred) # np.mean(np.array([augmented_pred[i,:][augmented_idx[i,:]] for i in range(augmented_idx.shape[0])]),axis=0)
                                         
                val_diff += phi_left - phi_right
        diff.append(1/binom(d-1,len(feature_subset)) * val_diff)
    shapley_j = 1/d*sum(diff)

    return shapley_j 
