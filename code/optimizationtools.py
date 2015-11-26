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
import pdb

# Optimization options
DISP=False
MAX_ITER=1


def optimize(fobj, f, theta, data, bnds=None, options=None):
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
        options (dictionary)
            grad (Bool):
                True if using gradient for optimization, else False
            pathway (int):
                LNK model pathway (1, 2, otherwise None)

    Output
    ------
        result
            theta: estimated parameter
            success: optimization converged
            fun: objective value
            corrcoef: correlation coefficient
            evar: explained variance
            theta_init: initial theta
            jac: gradient
    '''

    theta_init = theta

    if bnds:
        if options['is_grad']:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp':DISP, 'maxiter':MAX_ITER})
        else:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B',bounds=bnds, options={'disp':DISP, 'maxiter':MAX_ITER})
    else:
        # optimization
        if options['is_grad']:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B', jac=True, options={'disp':DISP, 'maxiter':MAX_ITER})
        else:
            res = minimize(fobj, theta_init, args=data, method='L-BFGS-B', options={'disp':DISP, 'maxiter':MAX_ITER})


    model = options['model']
    if model.lower() == 'lnks_mp':
        y = data[1][1]
        v, y_est = f(theta, data[0], options['pathway'])
    else:
        y = data[1]
        y_est = f(theta, data[0], options['pathway'])

    cc = _stats.corrcoef(y_est, y)
    ev = _stats.variance_explained(y_est, y)

    # results
    if options['is_grad']:
        jac = res.jac
    else:
        jac = None

    result = {
        "theta": res.x,
        "success": res.success,
        "fun": res.fun,
        "corrcoef": cc,
        "evar": ev,
        "theta_init": theta_init,
        "jac": jac}

    return result
