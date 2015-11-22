'''
optimizationtools.py

Tools for optimizing LNKS, LNK, Spiking models

author: Bongsoo Suh
created: 2015-02-26

(C) 2015 bongsoos
'''

import numpy as _np
import statstools as _stats
from scipy.optimize import minimize
from scipy.stats import pearsonr

# Optimization options
DISP=False
MAX_ITER=1


def optimize(fobj, f, theta, data, bnds=None, grad=False, pathway=None):
    '''
    Optimization using scipy.optimize.minimize module

    Input
    -----
        fobj (function):
            objective function
        f (function):
            model function
        theta (ndarray):
            initial parameter
        data (tuple of ndarray):
            input and output data
        bnds (tuple of tuples or None):
            Boundaries of model parameters
        grad (Bool):
            True if using gradient for optimization, else False
        pathway (int):
            LNK model pathway (1, 2, otherwise None)
    '''

    theta_init = theta

    if bnds:
        if grad:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp':DISP, 'maxiter':MAX_ITER})
        else:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B',bounds=bnds, options={'disp':DISP, 'maxiter':MAX_ITER})
    else:
        # optimization
        if grad:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B', jac=True, options={'disp':DISP, 'maxiter':MAX_ITER})
        else:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B', options={'disp':DISP, 'maxiter':MAX_ITER})


    # correlation
    if pathway:
        y_est = f(res.x, data[0], pathway=pathway)
        cc = _stats.corrcoef(y_est, data[1])

    else:
        y_est = f(res.x, data[0])
        cc = _stats.corrcoef(y_est, data[1])

    # results
    if grad:
        jac = res.jac
    else:
        jac = None

    result = {
        "theta": res.x,
        "success": res.success,
        "fun": res.fun,
        "corrcoef": cc,
        "theta_init": theta_init,
        "jac": jac}

    return result
