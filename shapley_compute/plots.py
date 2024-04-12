""" Created by : Claire He 
    12.04.24

    Generate minipatches 
functions: 
    - visualise_minipatch
              
"""

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


def bar(shap_values):
    