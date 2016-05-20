'''
statstools.py

Statistical tools for analyzing data

author: Bongsoo Suh
created: 2015-03-08

(C) 2015 bongsoos
'''

import numpy as _np


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

    X (numpy array): mxn data matrix(each column feature, each row is one data point)
    '''

    mu = _np.array([list(_np.mean(X, axis=0))]*X.shape[0])
    B = X-mu

    c = _np.dot(B.T, B) / (X.shape[0]-1)

    return c
