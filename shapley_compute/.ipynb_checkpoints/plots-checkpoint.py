""" Created by : Claire He 
    12.04.24

    Generate minipatches 
functions: 
    - visualise_minipatch
              
"""
import seaborn as sns
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

palette = sns.color_palette([
    "#7fbf7b",  # Light Green
    "#af8dc3",  # Lavender
    "#e7d4e8",  # Light Purple
    "#fdc086",  # Light Orange
    "#ff9896",  # Light Red
    "#c5b0d5"   # Light Blue
])
def visualise_minipatch(in_mp_obs, in_mp_feature, color_palette = palette, type='sorted'):
    
    B = in_mp_obs.shape[0]
    matrix = np.zeros((in_mp_obs.shape[1],in_mp_feature.shape[1]))
    for i in range(B):
        matrix += (in_mp_obs[i][:, np.newaxis] & in_mp_feature[i]).astype(int)
    # df = pd.DataFrame(matrix, columns =['feature_{}'.format(i) for i in range(in_mp_feature.shape[1])])
    if type =='sorted':
        sorted_M = np.sort(matrix, axis=0)
        sns.heatmap(np.sort(sorted_M,axis=1), cmap=color_palette)
        plt.yticks([])
    else:
        sns.heatmap(matrix, cmap=color_palette)
    plt.title('Patch selection coverage (permuted rows)')

# waterfall, beeswarm, barplots
    