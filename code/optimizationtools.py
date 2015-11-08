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
    #J1 = log_fobj(f, theta, x_in, y)
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
    J = _np.zeros(num_section)
    for i in range(num_section):
        # data range
        if i == 0:
            x_range = _np.arange(1000,len_section*(i+1))
        else:
            x_range = _np.arange(i*len_section, (i+1)*len_section)

        # weight
        if weight_type == "std":
            weight = _np.std(y[x_range]) + 1e-6
        elif weight_type == "mean":
            weight = _np.mean(y[x_range]) + 1e-3

        J[i] = mse_loss(y[x_range], y_est[x_range]) * (weight**2)


    return _np.sum(J)


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
    J = _np.zeros(num_section)
    for i in range(num_section):
        # data range
        if i == 0:
            x_range = _np.arange(1000,len_section*(i+1))
        else:
            x_range = _np.arange(i*len_section, (i+1)*len_section)

        # weight
        if weight_type == "std":
            weight = _np.std(y[x_range]) + 1e-6
        elif weight_type == "mean":
            weight = _np.mean(y[x_range]) + 1e-3

        J[i] = poisson_loss(y[x_range], y_est[x_range]) / (weight**2)

    return _np.sum(J)


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
    temp = _np.log(y_est)
    temp[_np.isinf(temp)] = -1e-6
    J = _np.sum(y_est - y*temp)

    return J



def fobj_numel_grad(fobj, f, theta, x_in, y, **kwargs):
    '''
    numerical gradient
    :param fobj:
    :param f:
    :param theta:
    :param x_in:
    :param y:
    :return:
    '''
    if kwargs is not None:
        pathway = kwargs["path"]
        J0 = fobj(f, theta, x_in, y, pathway) # evaluate function value at original point
    else:
        pathway = None
        J0 = fobj(f, theta, x_in, y) # evaluate function value at original point

    grad = _np.zeros(theta.shape)
    h = 0.00001

    # iterate over all indexes in theta
    it = _np.nditer(theta, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at theta+h
        i0 = it.multi_index
        theta[i0] += h # increment by h

        # evaluate f(theta+h)
        if pathway:
            Jh = fobj(f, theta, x_in, y, pathway)
        else:
            Jh = fobj(f, theta, x_in, y)

        theta[i0] -= h # restore to previous value (very important!)

        # compute the partial derivative
        grad[i0] = (Jh - J0) / h # the slope
        it.iternext() # step to next dimension

    return grad

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
        #print(i0, grad[i0], g0[i0])
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
    h = 1e-5

    thetas = _np.random.rand(theta.size, num_checks)
    thetas = (thetas - 0.5) * 10
    rel_error = _np.zeros(thetas.shape)
    abs_error = _np.zeros(thetas.shape)

    for i in range(num_checks):

        theta = thetas[:,i]
        rel_error[:,i], abs_error[:,i] = grad_check_single(f, theta, x_in, y)
        #ng = eval_numerical_gradient(f, theta, x_in, y)
        #J, ag = f(theta, x_in, y)

        #print(J)

        # rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        #rel_error[:,i] = _np.abs(ng - ag) / (_np.abs(ng) + _np.abs(ag)) * 100
        #abs_error[:,i] = _np.abs(ng - ag)

    print('Average relative error: %e' % (_np.mean(rel_error)))
    print('Average absolute error: %e' % (_np.mean(abs_error)))



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



def optimize(fobj, f, theta, data, bnds=None, grad=False, num_trials=1):
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

        if bnds == None:
            # optimization
            if grad:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B', jac=True, options={'disp':True})
            else:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B', options={'disp':True})
        else:
            if grad:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B', jac=True, bounds=bnds(), options={'disp':True})
            else:
                res = minimize(fobj, theta0, args=data, method='L-BFGS-B',bounds=bnds(), options={'disp':True})


        # initial costs
        if num_trials > 1:
            theta_inits[i,:] = theta0
        else:
            theta_inits = theta0

        #ccs_inits[i] = pearsonr(f(theta0, data[0]), data[1])[0]
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



def corrcoef(x, y):
    '''
    computes correlation coefficient (pearson's r)
    '''

    V = _np.sqrt(_np.array([_np.var(x, ddof=1), _np.var(y, ddof=1)])).reshape(1, -1)
    cc = (_np.matrix(_np.cov(x, y)) / (V*V.T + 1e-12))

    return cc[0,1]
