'''
objectivetools.py

Objective functions for LNKS, LNK, Spiking model optimization,
and tools for computing gradients

author: Bongsoo Suh
created: 2015-02-26

(C) 2015 bongsoos
'''

import numpy as _np
import time


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
    numerical gradient
    :param fobj:
    :param f:
    :param theta:
    :param x_in:
    :param y:
    :return:
    '''

    h = 0.00001
    thetas = perturbed_thetas(theta, h)
    thetas.append(theta)

    Jhs = [fobj(f, thetas[i], x_in, y, options) for i in range(len(thetas))]
    J0s = _np.array([Jhs[-1]] * theta.size)
    Jhs.pop()

    grad = (_np.array(Jhs) - J0s)/h

    return grad

def perturbed_thetas(theta, h):
    return [add(theta, h, i) for i in range(theta.size)]

def add(theta, h, i):
    temp = _np.zeros(theta.size)
    temp[i] = h
    return (theta + temp)

def eval_numerical_gradient(f, theta, x_in, y):
    """
    a naive implementation of numerical gradient of f at theta

    Input
    -----
        f (function):
            The objective function that returns cost value J, and the analytic gradient g0.
        theta (ndarray):
            The point (numpy array) to evaluate the gradient at
        x_in (ndarray):
            The point (numpy array) to evaluate the gradient at
        y (ndarray):
            The real data

    Output
    ------
        grad (ndarray):
            The numerical gradient
    """

    J0, g0 = f(theta, x_in, y) # evaluate function value at original point
    grad = _np.zeros(theta.shape)
    h = 0.00001

    # iterate over all indexes in theta
    it = _np.nditer(theta, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at theta+h
        i0 = it.multi_index
        theta[i0] += h # increment by h
        Jh, gh = f(theta, x_in, y) # evalute f(theta + h)
        theta[i0] -= h # restore to previous value (very important!)

        # compute the partial derivative
        grad[i0] = (Jh - J0) / h # the slope
        # print(i0, grad[i0], g0[i0])
        it.iternext() # step to next dimension

    return grad

def grad_check_single(f, theta, x_in, y):
    """
    Check gradients at multiple points


    Input
    -----
        f (function):
            The objective function that returns cost value J, and the analytic gradient g0.
        theta (ndarray):
            The point (numpy array) to evaluate the gradient at
        x_in (ndarray):
            The point (numpy array) to evaluate the gradient at
        y (ndarray):
            The real data

    Output
    ------
    rel_error (double):
        relative error
    abs_error (double):
        absolute error
    """
    ng = eval_numerical_gradient(f, theta, x_in, y)
    J, ag = f(theta, x_in, y)

    rel_error = _np.abs(ng - ag) / (_np.abs(ng) + _np.abs(ag)) * 100
    abs_error = _np.abs(ng - ag)

    return rel_error, abs_error


def grad_check_multi(f, theta, x_in, y, num_checks):
    """
    Check gradients at multiple points


    Input
    -----
        f (function):
            The objective function that returns cost value J, and the analytic gradient g0.
        theta (ndarray):
            The point (numpy array) to evaluate the gradient at
        x_in (ndarray):
            The point (numpy array) to evaluate the gradient at
        y (ndarray):
            The real data

    Output
    ------
    """
    thetas = _np.random.rand(theta.size, num_checks)
    thetas = (thetas - 0.5) * 10
    rel_error = _np.zeros(thetas.shape)
    abs_error = _np.zeros(thetas.shape)

    for i in range(num_checks):

        theta = thetas[:,i]
        rel_error[:,i], abs_error[:,i] = grad_check_single(f, theta, x_in, y)
        # ng = eval_numerical_gradient(f, theta, x_in, y)
        # J, ag = f(theta, x_in, y)

        # print(J)

        # rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        # rel_error[:,i] = _np.abs(ng - ag) / (_np.abs(ng) + _np.abs(ag)) * 100
        # abs_error[:,i] = _np.abs(ng - ag)

    print('Average relative error: %e' % (_np.mean(rel_error)))
    print('Average absolute error: %e' % (_np.mean(abs_error)))



