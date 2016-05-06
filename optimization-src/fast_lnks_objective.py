#!/usr/bin/env python3

'''
fast_lnks_objective.py

Fast LNKS model objective function for optimzing LNK, LNKS, LNKS_MP models.
Uses multiprocessing python module for parallel processing.

author: Bongsoo Suh
created: 2015-12-04

(C) 2015 bongsoos
'''

import numpy as _np
from scipy.linalg import orth as _orth
import lnkstools as _lnks
import kineticblocks as _kb
import spikingblocks as _sb
import objectivetools as _obj
import time as _time
import multiprocessing as _mult
import pdb as _pdb


def LNKS_fobj(theta, stim, y, options):
    '''
    LNKS Objective function
    -----------------------
    This is an objective function for the LNK, LNKS, and LNKS_MP models.
    Calls LNKS_fobj_helper function, and numel_gradient function(objectivetools),
    if is_grad option is True. The numel_gradient function uses multiprocessing
    , a built-in python module, to make the gradient computing process parallel.

    Returns objective value(J) and gradient(grad).
    If is_grad option is False, only returns objective value.

    Inputs
    ------
        theta: model parameters
        stim: input data
        y: output data (fr)
        options (dictionary):
            model (string): models are ('LNK', 'LNKS', 'LNKS_MP')
            pathway (int): LNK pathway (1 or 2)
            is_grad (bool): bool (gradient on(True) or off(False))

    Outputs
    -------
        J: objective value
        grad: gradient of objective
    '''

    basis = LinearFilterBasis_8param()
    nzstim = stim - _np.mean(stim) # nzstim: mean subtracted stimulus
    options['basis'] = basis

    J0 = LNKS_fobj_helper(LNKS, theta, nzstim, y, options) # evaluate function value at original point

    if options['is_grad']:
        grad = _obj.numel_gradient(LNKS_fobj_helper, LNKS, theta, nzstim, y, J0, options)
        return J0, grad

    else:
        return J0

def LNKS_fobj_helper(f, theta, stim, y, options):
    '''
    LNKS model objective function helper function

    Weighted sum of log-likelihood and mean-square error
    '''

    model = options['model']
    gamma = options['gamma']
    beta = options['beta']
    len_section=20000

    if model.lower() == 'lnk':
        y_est = f(theta, stim, options)
        J = _obj.mse_weighted_loss(y, y_est, len_section=len_section, weight_type='std')

    elif model.lower() == 'lnk_fr':
        y_est = f(theta, stim, options)
        J = _obj.mse_weighted_loss(y, y_est, len_section=len_section, weight_type='none')

    elif model.lower() == 'lnks':
        y_est = f(theta, stim, options)
        # linear combination of objective functions
        J_mse = _obj.mse_weighted_loss(y, y_est, len_section=len_section, weight_type='mean')
        J_poss = _obj.poisson_weighted_loss(y, y_est, len_section=len_section, weight_type="mean")
        J = (1-beta) * J_mse + beta * J_poss

    elif model.lower() in ['lnks_mp', 'lnks_spike']:
        # data
        y_mp = y[0]
        y_fr = y[1]
        y_mp_est, y_fr_est = f(theta, stim, options)
        # linear combination of objective functions
        if model.lower() == 'lnks_mp':
            J_mp = _obj.mse_weighted_loss(y_mp, y_mp_est, len_section=len_section, weight_type="std")
        elif model.lower() == 'lnks_spike':
            J_mp = _obj.mse_weighted_loss(y_mp, y_mp_est, len_section=len_section, weight_type="none")
        J_fr_poss = _obj.poisson_weighted_loss(y_fr, y_fr_est, len_section=len_section, weight_type="mean")
        J_fr_mse = _obj.mse_weighted_loss(y_fr, y_fr_est, len_section=len_section, weight_type="mean")
        J_fr = (1-beta) * J_fr_mse + beta * J_fr_poss
        J = (1-gamma) * J_mp + gamma * J_fr

        # print('J_fr_poss: %10.2f, J_fr_mse: %10.2f, J_mp: %10.2f' %(0.25*J_fr_poss, 0.25*J_fr_mse, 0.5*J_mp))

    return J


def LNKS(theta, stim, options):

    pathway = options['pathway']
    model = options['model']
    basis = options['basis']

    # one pathway
    if pathway == 1:
        thetas = [theta[:17]]
        weights = _np.array([theta[17]], dtype=_np.float64)
        if model.lower() in ['lnks', 'lnks_mp', 'lnks_spike']:
            theta_S = theta[18:]

    # two pathway
    elif pathway == 2:
        # first half: on pathway
        # second half: off pathway
        theta_on = theta[:17]
        theta_off = theta[18:35]
        thetas = [theta_on, theta_off]
        if model.lower() in ['lnks', 'lnks_mp', 'lnks_spike']:
            theta_S = theta[36:]
        # weights
        w_on = theta[17]
        w_off = theta[35]
        weights = _np.array([w_on, w_off], dtype=_np.float64)
    else:
        raise ValueError('The pathway parameter should be 1 or 2')

    # Compute the linear filter, find the filtered output and nonlinearity
    f = [basis.dot(thetas[i][:basis.shape[1]]) for i in range(pathway)]
    g_temp = [_np.convolve(stim,f[i]) for i in range(pathway)]
    g = [g_temp[i][:stim.size] for i in range(pathway)]
    u = [sigmoid(thetas[i][basis.shape[1]] + thetas[i][basis.shape[1]+1] * g[i]) for i in range(pathway)]

    # Compute Kinetics block operation.
    thetaK = [_np.array(thetas[i][basis.shape[1]+2:]) for i in range(pathway)]
    X0 = _np.array([0.1,0.2,0.7,99]) # Initial Kinetics states
    X = [_kb.K4S_C(thetaK[i], X0, u[i]) for i in range(pathway)]
    v_temp = [_np.array(X[i][1,:], dtype=_np.float64) for i in range(pathway)]
    # v_temp = [v_temp[i] - _np.min(v_temp[i]) for i in range(pathway)]
    # v_temp = [v_temp[i]/ _np.max(v_temp[i]) for i in range(pathway)]

    # linear combination of pathway outputs
    output = sum([weights[i] * v_temp[i] for i in range(pathway)])
    output = output - _np.mean(output)

    if model.lower() == 'lnks':
        # Comptue Spiking model
        # spiking parameters following after LNK parameters
        output = _sb.SC1DF_C(theta_S, output)

        return output
    elif model.lower() in ['lnks_mp','lnks_spike']:
        # Comptue Spiking model
        # spiking parameters following after LNK parameters
        r = _sb.SC1DF_C(theta_S, output)

        # return both membrane potential and firing rate
        return output, r

    else:
        return output


def sigmoid(x):
    '''
    Return applied sigmoidal function on the input signal.

    Input
    -----
    x (ndarray):
        The 1-by-n dimensional array.

    Output
    ------
    y (ndarray):
        The sigmoidal output.

    '''
    isexcept = x < -500
    if any(isexcept):
        # exception: exp creating Runtimewarning (Inf)
        y = _np.zeros(x.size)
        idx = _np.where(~isexcept)
        y[idx] = 1 / (1 + _np.exp(-x[idx]))
        return y
    else:
        return 1 / (1 + _np.exp(-x))


def LinearFilterBasis_15param():
    '''
    Generate 15 linear filter basis.

    Output
    ------
    basis (ndarray):
        The linear filter basis (1000 by 15)
    '''
    t = _np.linspace(0.001, 1, 1000)
    t2 = t*2 - t**2
    m1 = _np.outer(t2, _np.ones(15))
    m2 = _np.outer(_np.ones(1000), _np.arange(1,16))
    A = _np.sin(m1 * m2 * _np.pi)
    basis = _orth(A)

    return basis


def LinearFilterBasis_8param():
    '''
    Generate 8 linear filter basis in the range of 500ms.

    Output
    ------
    basis (ndarray):
        The linear filter basis (1000 by 8)
    '''

    basis = LinearFilterBasis_15param()

    return basis[:500,7:]

