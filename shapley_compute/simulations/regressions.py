""" Created by : Claire He 
    24.04.24

    Helper functions for simulation designs 
              
"""


import numpy as np



def normal_linear_model(N, M, SNR = 10, sigma2=0.4, s=0.2):
    """
    Simulates normal linear model with SNR 
    N: number of obs
    M: number of features
    sigma2: variance 
    s: sparsity level
    """ 
    np.random.seed(123)
    X = np.random.normal(0,1, size=(N,M))
    M1 = int(s*M)
    mu = np.sqrt(sigma2)*SNR
    beta = np.append(np.random.normal(mu, 1, M1),np.array([0]*(M-M1))) # M-M1 beta are set to 0, M1 are non zeros
    eps = np.random.normal(0, sigma2, size=N)

    y = X@beta + eps
    return y, X, beta

def toeplitz_covariance(rho, n):
    """ rho : correlation coefficient
        n : shape 
    """
    j, k = np.indices((n, n))
    Sigma = rho**np.abs(j - k)
    return Sigma


def correlated_features_regression(N, M, SNR = 10, sigma2=0.4, s=0.2, type='toeplitz', rho=0.5):
    """ Simulates structured linear regression with design Toeplitz or design equi-correlated matrix """
    np.random.seed(123)
    if type == 'toeplitz':
        X = np.random.multivariate_normal(np.zeros(M), toeplitz_covariance(rho, M), N)
    elif type == 'equi-corr':
        cov = (1-rho)*np.eye(M)+ rho*np.ones((M,M))
        X = np.random.multivariate_normal(np.zeros(M), cov, N)
    M1 = int(s*M)
    mu = np.sqrt(sigma2)*SNR
    beta = np.append(np.random.normal(mu, 1, M1),np.array([0]*(M-M1))) # M-M1 beta are set to 0, M1 are non zeros
    eps = np.random.normal(0, sigma2, size=N)

    y = X@beta + eps
    return y, X, beta

