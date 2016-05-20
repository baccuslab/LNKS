'''
statstools.py

Statistical tools for analyzing data

author: Bongsoo Suh
created: 2015-03-08

(C) 2015 bongsoos
'''

import numpy as _np
from numpy.linalg import _eig

def corrcoef(x, y):
    '''
    computes correlation coefficient (pearson's r)
    '''

    V = _np.sqrt(_np.array([_np.var(x, ddof=1), _np.var(y, ddof=1)])).reshape(1, -1)
    cc = (_np.matrix(_np.cov(x, y)) / (V*V.T + 1e-12))

    return cc[0,1]


def variance_explained(y, y_est):
    '''
    Computes explained variance
    '''
    SYY = sum((y - _np.mean(y))**2)
    RSS = sum((y - y_est)**2)

    R2 = 1 - RSS/SYY

    return R2

def cov(X):
    '''
    Compute covariance matrix

    mean subtraction, and variance normalization
    X (numpy array): mxn data matrix(each column feature, each row is one data point)
    '''

    mu = _np.array([list(_np.mean(X, axis=0))]*X.shape[0])
    B = X-mu
    sigma = _np.sqrt(_np.var(B, axis=0))
    Bn = B/_np.outer(_np.ones(B.shape[0]), sigma)
    c = _np.dot(Bn.T, Bn) / (Bn.shape[0]-1)

    return c

def pca(X):
    d, v = _eig(cov(X))
    return v, d
