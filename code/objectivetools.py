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

def fobjective_numel(fobj, f, theta, data):
    '''
    Objective function of any model and objective. Computes its numerical gradient.

    Returns the cost value(J) and the gradient(grad).

    Input
    -----
    fobj (function):
        objective function
    f (function):
        model function
    theta (ndarray)
    data (tuple (ndarray,ndarray)):
        The data (input, output) tuple.

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input

    grad (ndarray):
        The gradient of the objective function
    '''
    J = fobj(f, theta, data[0], data[1])
    grad = fobj_numel_grad(fobj, f, theta, data[0], data[1])

    return J, grad


def log_diff_fobj(f, theta, x_in, y):
    '''
    sum of weighted log-likelihood and mse
    '''
    J1 = log_fobj_weighted(f, theta, x_in, y)
    J2 = diff_fobj(f, theta, x_in, y)

    J = J1 + J2

    return J


def log_fobj(f, theta, x_in, y):
    '''
    Objective function of any model using log-likelihood(poisson_loss function)
    Returns the cost value(J).

    Input
    -----
    x_in (ndarray)
    theta (ndarray)
    y (ndarray):
        The firing rate data.

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input

    grad (ndarray):
        The gradient of the objective function
    '''

    y_est = f(theta, x_in)
    J = poisson_loss(y, y_est)

    return J



def log_fobj_weighted(f, theta, x_in, y):
    '''
    Objective function of any model using log-likelihood weighted sum

    Returns the cost value(J).

    Input
    -----
    x_in (ndarray)
    theta (ndarray)
    y (ndarray): The firing rate data.

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input

    grad (ndarray):
        The gradient of the objective function
    '''

    y_est = f(theta, x_in)

    J = poisson_weighted_loss(y, y_est, len_section=10000, weight_type="mean")

    return J


def diff_fobj(f, theta, x_in, y):
    '''
    Objective function that computes the weighted sum of Residual Sum of Squares for different contrast sections.

    Input
    -----
    x_in (ndarray)
    theta (ndarray)
    y (ndarray):
        The firing rate data.

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input

    grad (ndarray):
        The gradient of the objective function
    '''
    y_est = f(theta, x_in)

    J = mse_weighted_loss(y, y_est, len_section=10000, weight_type="mean")

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

    num_section = _np.int(_np.floor(y.size / len_section))

    y_list = [y[_np.arange(i*len_section, (i+1)*len_section)] for i in range(num_section)]
    y_est_list = [y_est[_np.arange(i*len_section, (i+1)*len_section)] for i in range(num_section)]

    # weights
    if weight_type == "std":
        J = _np.sum(_np.array([mse_loss(y_list[i], y_est_list[i])/((_np.std(y_list[i])+1e-6)**2) for i in range(len(y_list))]))

    elif weight_type == "mean":
        J = _np.sum(_np.array([mse_loss(y_list[i], y_est_list[i])/((_np.mean(y_list[i])+1e-3)**2) for i in range(len(y_list))]))

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
    x_in (ndarray)
    theta (ndarray)
    y (ndarray):
        The firing rate data.

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input
    '''

    num_section = _np.int(_np.floor(y.size / len_section))

    y_list = [y[_np.arange(i*len_section, (i+1)*len_section)] for i in range(num_section)]
    y_est_list = [y_est[_np.arange(i*len_section, (i+1)*len_section)] for i in range(num_section)]

    # weights
    if weight_type == "std":
        weights = _np.array(list(map(_np.std, y_list))) + 1e-6
    elif weight_type == "mean":
        weights = _np.array(list(map(_np.mean, y_list))) + 1e-3

    J = _np.sum(_np.array(list(map(poisson_loss, y_list, y_est_list))) / (weights**2))

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



def logistic_fobj(theta, X, y):
    '''
    Return objective function value and its gradient of logistic function
    using log-likelihood objective function and its gradient of logistic model.
    Returns the cost value(J) and the gradient(grad).

    Input
    -----
    X (ndarray)
    theta (ndarray)
    y (ndarray):
        The output data.

    Output
    ------
    J (double):
        The cost value measured by the likelihood function at given theta and the input

    grad (ndarray):
        The gradient of the objective function
    '''

    y_est = sigmoid(theta.dot(X))

    '''
    likelihood objective function
    '''
    temp = _np.log(y_est)
    temp[_np.isinf(temp)] = -1e-6
    J = _np.sum(y_est - y*temp)

    '''
    gradient of the objective function
    '''
    e = y_est - y
    w = _np.ones(y_est.shape) - y_est
    grad = _np.sum(e * w * X,1)

    return J, grad



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
    return 1 / (1 + _np.exp(-x))



