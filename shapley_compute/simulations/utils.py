""" Created by : Claire He 
    24.04.24

    Helper functions for computing target estimand
              
"""


import numpy as np
from sklearn.model_selection import LeaveOneOut
import tqdm

def get_LOO_predictions(X, y, model):
    loo = LeaveOneOut()
    pred_loo = []
    for train_index, test_index in loo.split(X):
        # Split the data into training and testing sets
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store the prediction
        pred_loo.append(y_pred[0])
    pred_loo = np.array(pred_loo) 
    return pred_loo

def MCDelta(X, y, model, n_iter=100):
    Delta = np.zeros(X.shape)
    for i in tqdm.tqdm(range(n_iter)):
        for j in range(X.shape[1]):
            ### Err(Y, mu_j(X_j))
            Xj = np.delete(X, j, axis=1)
            mu_j = get_LOO_predictions(Xj, y, model)
            err_j = np.abs(y - mu_j)
            
            ### Err(Y, mu(X))
            mu = get_LOO_predictions(X, y, model)
            err = np.abs( y - mu)
            
            Delta[:,j] += err_j - err
    return Delta/n_iter