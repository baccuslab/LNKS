'''
objectivetools.py

Objective functions for LNKS, LNK, Spiking model optimization,
and tools for computing gradients

author: Bongsoo Suh
created: 2015-02-26

(C) 2015 bongsoos
'''

import numpy as _np
import time as _time
import multiprocessing as _mult


def weighted_loss(loss_func, y, y_est, len_section=10000, weight_type="std"):
    '''
    Objective function of any model using any loss function
    and weighted sum of its sections.

    Returns the cost value(J).

    Input
    -----
    loss_func (function object): loss function with weight
    y (ndarray): The data.
    y_est (ndarray): The estimate.
    len_section (int): length of one section
    weight_type (string): type of weight ("std" or "mean")

    Output
    ------
    J (double):
        The cost value measured by the loss function at given theta and the input
    '''

    num_section = _np.int(_np.floor(y.size / len_section))

    y_list = [y[_np.arange(i*len_section, (i+1)*len_section)] for i in range(num_section)]
    y_est_list = [y_est[_np.arange(i*len_section, (i+1)*len_section)] for i in range(num_section)]

    J = _np.sum(_np.array([loss_func(y_list[i], y_est_list[i], weight_type) for i in range(num_section)]))
    return J


def mse_weighted_loss(y, y_est, len_section=10000, weight_type="std"):
    '''
    Weighted MSE(Residual Sum of Squares) objective function weighted by 1/C for each contrast section
    where C can be the mean or standard deviation of the section chosen by the `weight_type` variable.
    Calls `mse_loss` function.

    Inputs
    ------
        y (ndarray): output data
        y_est (ndarray): model output
        weight_type (string): "std" or "mean"

    Outputs
    -------
        J (double): objective value
    '''

    J = weighted_loss(mse_loss_helper, y, y_est, len_section, weight_type)

    return J


def mse_loss_helper(y, y_est, weight_type="std"):
    '''
    Objective function using poisson_loss.

    Returns weighted poisson loss cost value, where the weight depends on the weight_type.
    '''
    if weight_type == "std":
        weight = _np.std(y) + 1e-6
    elif weight_type == "mean":
        weight = _np.mean(y) + 1e-3
    elif weight_type == "none":
        weight = 1

    J = mse_loss(y, y_est) / (weight**2)

    return J


def mse_loss(y, y_est):
    '''
    MSE(Residual Sum of Squares) objective function

    Inputs
    ------
        y (ndarray): output data
        y_est (ndarray): model output

    Outputs
    -------
        J (double): objective value
    '''

    # error(residual) term
    err = y - y_est
    J = _np.sum((err ** 2))

    return J


def poisson_weighted_loss(y, y_est, len_section=10000, weight_type="std"):
    '''
    Objective function of any model using log-likelihood poisson loss function
    and weighted sum of sections.

    Returns the cost value(J).

    Input
    -----
    y (ndarray): The data.
    y_est (ndarray): The estimate.
    len_section (int): length of one section
    weight_type (string): type of weight ("std" or "mean")

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input
    '''

    J = weighted_loss(poisson_loss_helper, y, y_est, len_section, weight_type)

    return J


def poisson_loss_helper(y, y_est, weight_type="std"):
    '''
    Objective function using poisson_loss.

    Returns weighted poisson loss cost value, where the weight depends on the weight_type.
    '''
    if weight_type == "std":
        weight = _np.std(y) + 1e-6
    elif weight_type == "mean":
        weight = _np.mean(y) + 1e-3

    J = poisson_loss(y, y_est) / (weight**2)

    return J


def poisson_loss(y, y_est):
    '''
    Objective function using log-likelihood under Poisson probability distribution assumption of the output y.
    Returns the cost value(J).

    Input
    -----
    y (ndarray): output data
    y_est (ndarray): model output

    Output
    ------
    J (double): The cost value measured by the likelihood function at given theta and the input

    '''

    # likelihood objective function
    eps = 1e-6
    temp = _np.log(y_est+eps)
    temp[_np.isinf(temp)] = -eps
    J = _np.sum(y_est - y*temp)

    return J



def fobj_numel_grad(fobj, f, theta, x_in, y, options):
    '''
    Returns numerical gradient
    Original version of the gradient using list comprehension.

    Input
    -----
        fobj (function object): objective function
        f (function object): model function
        theta (ndarray): model parameter
        x_in (ndarry): input data
        y (ndarray): output data
        options (dictionary):

    Output
    ------
        grad (ndarray)

    '''

    h = 0.00001
    thetas = perturbed_thetas(theta, h)
    thetas.append(theta)

    Jhs = [fobj(f, thetas[i], x_in, y, options) for i in range(len(thetas))]
    J0s = _np.array([Jhs[-1]] * theta.size)
    Jhs.pop()

    grad = (_np.array(Jhs) - J0s)/h

    return grad

def numel_gradient(fobj, f, theta, x_in, y, J0, options):
    '''
    Returns numerical gradient
    Uses multiprocessing python module for parallel computing using multiple threads of processes.

    Input
    -----
        fobj (function object): objective function
        f (function object): model function
        theta (ndarray): model parameter
        x_in (ndarry): input data
        y (ndarray): output data
        J0 (double): objective function value at point theta
        options (dictionary):
            model (string): models are ('LNK', 'LNKS', 'LNKS_MP')
            pathway (int): LNK pathway (1 or 2)
            is_grad (bool): bool (gradient on(True) or off(False))

    Output
    ------
        grad (ndarray)

    '''
    J0s = [J0]*theta.size
    h = 0.00001
    thetas = perturbed_thetas(theta, h)
    pool = _mult.Pool(processes=_mult.cpu_count())
    results = [pool.apply_async(fobj, args=(f, thetas[j], x_in, y, options)) for j in range(theta.size)]
    output = [p.get() for p in results]
    pool.close()

    Jhs = _np.array(output)
    grad = (Jhs - J0s)/h

    return grad


def perturbed_thetas(theta, h, mode='default'):
    if mode == 'default':
        return [add(theta, h, i) for i in range(theta.size)]
    elif mode == 'NKS':
        theta_range = _np.arange(8,theta.size)
        return [add(theta, h, i) for i in theta_range]

def add(theta, h, i):
    temp = _np.zeros(theta.size)
    temp[i] = h
    theta_perturbed = theta + temp
    return theta_perturbed

