'''
optimizationtools.py

Tools for optimizing LNKS, LNK, Spiking models

author: Bongsoo Suh
created: 2015-02-26

(C) 2015 bongsoos
'''

import numpy as _np
from scipy.optimize import minimize
from scipy.stats import pearsonr


def optimize(fobj, f, theta, data, bnds=None, grad=False, num_trials=1, pathway=None):
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
    '''

    theta_inits = theta
    if num_trials > 1:
        theta_inits = _np.zeros([num_trials, theta.size])
    thetas = _np.zeros([num_trials, theta.size])
    succs = []
    ccs = _np.zeros(num_trials)
    ccs_inits = _np.zeros(num_trials)
    funs = _np.zeros(num_trials)
    funs_inits = _np.zeros(num_trials)
    jacs = _np.zeros([num_trials,theta.size])


    # for all the initials, optimize the SCI model
    for i in range(num_trials):
        if num_trials > 1:
            theta0 = _np.array((_np.random.rand(theta.size)-0.5)*100)
        else:
            theta0 = theta

        disp=False
        max_iter=100
        if bnds:
            if grad:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B', jac=True, bounds=bnds, options={'disp':disp, 'maxiter':max_iter})
            else:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B',bounds=bnds, options={'disp':disp, 'maxiter':max_iter})
        else:
            # optimization
            if grad:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B', jac=True, options={'disp':disp, 'maxiter':max_iter})
            else:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B', options={'disp':disp, 'maxiter':max_iter})


        # initial costs
        if num_trials > 1:
            theta_inits[i,:] = theta0
        else:
            theta_inits = theta0

        ccs_inits[i] = corrcoef(f(theta0, data[0]), data[1])
        if grad:
            funs_inits[i], grad_init = fobj(theta0, data[0], data[1])
        else:
            funs_inits[i] = fobj(theta0, data[0], data[1])

        # results
        thetas[i,:] = res.x
        succs.append(res.success)
        funs[i] = res.fun
        if grad:
            jacs[i,:] = res.jac
        #ccs[i] = pearsonr(f(res.x, data[0]), data[1])[0]
        ccs[i] = corrcoef(f(res.x, data[0]), data[1])

        if num_trials > 1:
            if (i % 100 == 0):
                print("Optimization process %% %d completed..." % (i/num_trials * 100))

    if num_trials > 1:
        print("\nOptimization Finished \n")
        print("%d successes out of %d trials" % (sum(succs), num_trials))

        result = {"theta": thetas,
                "success": succs,
                "fun": funs,
                "corrcoef": ccs,
                "theta_inits": theta_inits,
                "fun_inits": funs_inits,
                "corrcoef_inits": ccs_inits,
                "jac": jacs}
    else:
        result = {"theta": thetas[0],
                "success": succs[0],
                "fun": funs[0],
                "corrcoef": ccs[0],
                "theta_inits": theta_inits[0],
                "fun_inits": funs_inits[0],
                "corrcoef_inits": ccs_inits[0],
                "jac": jacs[0]}

    return result


def optimize_cells(fobj, theta, types, model, num_trials=1):
    '''
    optimize cells
    :param fobj:
    :param theta:
    :param types:
    :param model:
    :param num_trials:
    :return:
    '''

    if types == "all":
        cells = [_ldt.g4(), _ldt.g6(), _ldt.g8(), _ldt.g9(), _ldt.g9apb(), _ldt.g11(), _ldt.g12(), _ldt.g13(), _ldt.g15(), _ldt.g17(), _ldt.g17apb(), _ldt.g19(), _ldt.g19apb(), _ldt.g20(), _ldt.g20apb()]
        fig = _plt.figure(figsize=(30,18))

    elif types == "off":
        cells = [_ldt.g9apb(), _ldt.g11(), _ldt.g17apb(), _ldt.g19apb(), _ldt.g20apb()]
        fig = _plt.figure(figsize=(30,6))

    elif types == "on-off":
        cells = [_ldt.g6(), _ldt.g8(), _ldt.g9(), _ldt.g13(), _ldt.g15(), _ldt.g17(), _ldt.g19(), _ldt.g20()]
        fig = _plt.figure(figsize=(24,12))

    elif types == "sens":
        cells = [_ldt.g13(), _ldt.g20()]
        fig = _plt.figure(figsize=(12,6))

    elif types == "reverse":
        cells = [_ldt.g4(), _ldt.g12()]
        fig = _plt.figure(figsize=(12,6))

    else:
        print("select types")

