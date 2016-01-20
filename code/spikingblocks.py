#!/usr/bin/env python3
'''
spikingblocks.py

Implementation of different Spiking Block topologies

author: Bongsoo Suh
created: 2015-02-24

(C) 2015 bongsoos
'''

import numpy as _np
import spikingtools as _st
from scipy.linalg import toeplitz as _tpltz
import objectivetools as _obj
from scipy import interpolate as interp
import analysistools as _at
import dataprocesstools as _dpt
import pdb


'''
    Spiking GLM model(SG)
'''
def SG(theta, x_in):
    '''
    Spiking GLM model.
    Compute the Spiking block using generalized linear model(GLM).
    Input is x, dx/dt, and history(firing rate).
    '''

    # dx_in = deriv(x_in, 0.001)
    # need to implement
    y = _np.zeros(x_in.shape)

    return y

'''
    Spiking GLM model(SG) objective functions, gradient, gain
'''
def SG_fobj(theta, x_in, y):
    '''
    Objective function and its gradient
    Likelihood objective function and its gradient of Spiking Continuous(SC) spiking model.
    see SC(x_in, theta).
    Returns the cost value(J) and the gradient(grad).

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

    # C = 0.001
    dx_in = deriv(x_in, 0.001)
    # filt_len = 1000

    X = _np.zeros([3, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in

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



'''
    Spiking Continuous 1-D (SC1D) model (1-D approximation of higher order firing rate model).
'''
def SC1D(theta, x_in, options):
    '''
    Spiking Continuous 1-D model.
    Compute the very basic continuous 1D nonlinearity(sigmoid) spiking block.
    The 1D nonlinearity is a function of x and dx/dt.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([3, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in

    y = sigmoid(theta.dot(X))

    return y


'''
    Spiking Continuous 1-D (SC1D) objective functions, gradient, gain
'''
def SC1D_fobj(theta, x_in, y, options):
    '''
    Objective function and its gradient
    Likelihood objective function and its gradient of Spiking Continuous(SC) spiking model.
    see SC(x_in, theta).
    Returns the cost value(J) and the gradient(grad).

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


    J0 = SC1DF_fobj_helper(SC1D, theta, x_in, y, options)

    if options['is_grad']:
        grad = _obj.numel_gradient(SC1DF_fobj_helper, SC1D, theta, x_in, y, J0, options)
        return J0, grad
    else:
        return J0

#
#    J, grad = _obj.fobjective_numel(_obj.log_diff_fobj, SC1D, theta, (x_in, y))
#
#    if False:
#        dx_in = deriv(x_in, 0.001)
#
#        X = _np.zeros([3, x_in.size])
#        X[0,:] = _np.ones(x_in.size)
#        X[1,:] = x_in
#        X[2,:] = dx_in
#
#        y_est = sigmoid(theta.dot(X))
#
#        '''
#        likelihood objective function
#        '''
#        temp = _np.log(y_est)
#        temp[_np.isinf(temp)] = -1e-6
#        J = _np.sum(y_est - y*temp)
#
#        '''
#        gradient of the objective function
#        '''
#        e = y_est - y
#        w = _np.ones(y_est.shape) - y_est
#        grad = _np.sum(e * w * X,1)
#
#    return J, grad

def SC1D_bnds(pathway=None):
    '''
    return SC1D bound constraints.
    '''
    return ((None,None),(None,None),(None,None))

def SC1D_1d(theta, x, dx):
    '''
    returns 2 dimensional array of SC1D model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''

    X = _np.array([1, x, dx])

    resp = sigmoid(theta.dot(X))

    return resp


def SC1D_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SCIE model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            x = VV[i,j]
            dx = dVV[i,j]
            resp[i,j] = SC1D_1d(theta, x, dx)

    return resp

def SC1D_gain(theta, x_in):
    '''
    Gain of the Spiking Continuous.
    Compute the gain of the SC spiking block model, which is the gradient at given input(and its derivative).

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    gain (ndarray):
        The gain of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([3, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in

    y_est = sigmoid(theta.dot(X))
    gain = y_est * (1 - y_est) * theta[1]

    return gain



'''
    Spiking Continuous 1-D Feedback (SC1DF) model (1-D approximation of higher order firing rate model).
'''
def SC1DF(theta, x_in, options=None):
    '''
    Spiking Continuous 1-D model.
    Compute the very basic continuous 1D nonlinearity(sigmoid) spiking block with feedback.
    The 1D nonlinearity is a function of x and dx/dt.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)
    # dx_in = deriv_backward(x_in, 0.001)

    y = _st.SC1DF(theta, x_in, dx_in)
    h = _st.SC1DF_get_h(theta, x_in, dx_in)
    m = _st.SC1DF_get_m(theta, x_in, dx_in)
    gain = _st.SC1DF_gain(theta, x_in, dx_in)

    b = h - x_in

    theta_fb = theta[3:]
    # len_fb = 5000
    len_fb = 1000
    t = _np.arange(len_fb)
    fb1 = theta_fb[0] * _np.exp( -t / theta_fb[1])
    fb2 = theta_fb[2] * _np.exp( -t / theta_fb[3])
    # fb3 = theta_fb[4] * _np.exp( -t / theta_fb[5])
    # fb = (fb1 + fb2)/2 + fb3
    fb = (fb1 + fb2)/2

    return y, h, gain, fb, b, m

def SC1DF_C(theta, x_in, options=None):
    '''
    Spiking Continuous 1-D model.
    Compute the very basic continuous 1D nonlinearity(sigmoid) spiking block.
    The 1D nonlinearity is a function of x and dx/dt.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)
    # dx_in = deriv_backward(x_in, 0.001)

    y = _st.SC1DF(theta, x_in, dx_in)

    return y


'''
    Spiking Continuous 1-D Feedback(SC1DF) objective functions, gradient, gain
'''
def SC1DF_fobj(theta, x_in, y, options):
    '''
    Objective function and its gradient
    Likelihood objective function and its gradient of Spiking Continuous(SC) spiking model.
    see SC(x_in, theta).
    Returns the cost value(J) and the gradient(grad).

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

    J0 = SC1DF_fobj_helper(SC1DF_C, theta, x_in, y, options)

    if options['is_grad']:
        grad = _obj.numel_gradient(SC1DF_fobj_helper, SC1DF_C, theta, x_in, y, J0, options)
        return J0, grad
    else:
        return J0


def SC1DF_fobj_helper(f, theta, stim, y, options):
    '''
    LNKS model objective function helper function

    Weighted sum of log-likelihood and mean-square error
    '''

    y_est = f(theta, stim, options['pathway'])

    # linear combination of objective functions
    J_poss = _obj.poisson_weighted_loss(y, y_est, len_section=10000, weight_type="mean")
    J_mse = _obj.mse_weighted_loss(y, y_est, len_section=10000, weight_type="mean")
    J = J_poss + J_mse

    return J

def SC1DF_bnds(theta=None, pathway=None, bnd_mode=0):
    '''
    return SCIF bound constraints.
    '''
    numFB = 2
    thr_bnds = ((None,None),(None,None),(None,None))
    fb_bnds = tuple([(0,None) for i in range(numFB*2)])

    bnds = thr_bnds + fb_bnds

    return bnds

def SC1DF_1d(theta, x, dx):
    '''
    returns 2 dimensional array of SC1DF model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''

    X = _np.array([1, x, dx])

    resp = sigmoid(theta.dot(X))

    return resp


def SC1DF_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SCIE model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            x = VV[i,j]
            dx = dVV[i,j]
            resp[i,j] = SC1DF_1d(theta, x, dx)

    return resp

def SC1DF_gain(theta, x_in):
    '''
    Gain of the Spiking Continuous.
    Compute the gain of the SC spiking block model, which is the gradient at given input(and its derivative).

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    gain (ndarray):
        The gain of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)
    gain = _st.SC1DF_gain(theta, x_in, dx_in)

    return gain


def SC1DF_constFB(theta, x_in):
    '''
    Spiking Continuous Independent Feedback with const Feedback(no effect of Feedback)
    '''

    dx_in = deriv(x_in, 0.001)
    h = _st.SC1DF_get_h(theta, x_in, dx_in)
    b = h - x_in
    mean_b = _np.mean(b)

    h_const = x_in + mean_b

    y = SC1D(theta[:3], h_const)
    gain = SC1D_gain(theta[:3], h_const)

    return y, gain


def SC1DF_Impulse(theta, x_in, delta=1, index=0, input_mode='data'):
    '''
    Impulse response of Spiking Continuous 1-D model.
    Compute from the analysis of instantaneous gain, the slope at delta is zero.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    delta (double):
        The impulse amplitude.

    index (int):
        The index where the impulse occurs.

    input_mode (string):
        The mode of finding the Impulse response, using different type of input.
        input_mode is selected from {"data", "impulse", "step"}.
            "data": x_in is a membrane potential(data or LNK output)
            "impulse": x_in is a zeros array
            "step": x_in is a zeros input

    Output
    ------
    r_impulse (ndarray):
        The impulse response
    '''
    infinite = 10000
    dx_in = deriv(x_in, 0.001)
    x_impulse = _np.zeros(len(x_in))

    if input_mode == 'data':
        x_impulse[index] = delta
        x_in_p = x_in + x_impulse

        y = _st.SC1DF(theta, x_in, dx_in)
        y_p = _st.SC1DF(theta, x_in_p, dx_in)

        r_impulse = y_p - y

    elif input_mode == 'impulse':
        x_impulse[index] = delta

        r_impulse = _st.SC1DF(theta, x_impulse, dx_in)

    elif input_mode == 'step':
        x_impulse[index:] = delta
        dx_impulse = _np.zeros(len(x_in))
        dx_impulse[index] = infinite
        dx_in_p = dx_in + dx_impulse

        r_impulse = _st.SC1DF(theta, x_impulse, dx_in_p)


    return r_impulse

'''
    Spiking Continuous(SC) models
'''
def SC(theta, x_in):
    '''
    Spiking Continuous.
    Compute the very basic continuous 1D nonlinearity(sigmoid) spiking block.
    The 1D nonlinearity is a function of x and dx/dt.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([6, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in
    X[3,:] = x_in * dx_in
    X[4,:] = x_in ** 2
    X[5,:] = dx_in ** 2

    y = sigmoid(theta.dot(X))

    return y

def SCm(theta, x_in):
    '''
    Spiking Continuous multi-feature.
    Compute the continuous 1D nonlinearity(sigmoid) spiking block.
    The 1D nonlinearity is a function of x and dx/dt.
    'm' stands for multi-features having higher order features, whereas SC1 having first order features.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([6, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in
    X[3,:] = x_in * dx_in
    X[4,:] = x_in ** 2
    X[5,:] = dx_in ** 2
    X[6,:] = (x_in ** 2) * (dx_in)
    X[7,:] = (x_in) * (dx_in ** 2)
    X[8,:] = (x_in ** 2) * (dx_in ** 2)
    X[9,:] = (x_in ** 3)
    X[10,:] = (dx_in ** 3)


    y = sigmoid(theta.dot(X))

    return y

def SC_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SC model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            X = _np.array([1, VV[i,j], dVV[i,j], VV[i,j]*dVV[i,j], VV[i,j]**2, dVV[i,j]**2])
            resp[i,j] = sigmoid(theta.dot(X))

    return resp

def SC1_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SC1 model's 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            X = _np.array([1, VV[i,j], dVV[i,j]])
            resp[i,j] = sigmoid(theta.dot(X))

    return resp

def SCm_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SCm model's 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            X = _np.array([1, VV[i,j]**2, VV[i,j]*dVV[i,j], dVV[i,j]**2])
            resp[i,j] = sigmoid(theta.dot(X))

    return resp

'''
    Spiking Continuous(SC) objective functions, gradient, gain
'''
def SC_fobj(theta, x_in, y):
    '''
    Objective function and its gradient
    Likelihood objective function and its gradient of Spiking Continuous(SC) spiking model.
    see SC(x_in, theta).
    Returns the cost value(J) and the gradient(grad).

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

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([6, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in
    X[3,:] = x_in * dx_in
    X[4,:] = x_in ** 2
    X[5,:] = dx_in ** 2

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


def SC_gain(theta, x_in):
    '''
    Gain of the Spiking Continuous.
    Compute the gain of the SC spiking block model, which is the gradient at given input(and its derivative).

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    gain (ndarray):
        The gain of the spiking block.
    '''

    # V_mat = _tpltz([x_in[0],0,0,0,0,0,0],x_in)
    c = _np.array([49/20, -6, 15/2, -20/3, 15/4, -6/5, 1/6])/0.001
    cg = c[0]

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([6, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in
    X[2,:] = dx_in
    X[3,:] = x_in * dx_in
    X[4,:] = x_in ** 2
    X[5,:] = dx_in ** 2

    C_mat = _np.zeros(X.shape)
    C_mat[1,:] = _np.ones(X.shape[1])
    C_mat[2,:] = _np.ones(X.shape[1]) * cg
    C_mat[3,:] = dx_in + cg * x_in
    C_mat[4,:] = 2 * x_in
    C_mat[5,:] = 2 * cg * dx_in

    y_est = sigmoid(theta.dot(X))
    gain = y_est * (1 - y_est) * (theta.dot(C_mat))

    return gain


'''
    Spiking Continuous Independent(SCI) models
'''
def SCI(theta, x_in):
    '''
    Spiking Continuous Independent.
    Compute basic continuous 2D nonlinearity spiking block.
    The 2D nonlinearity is a function of x and dx/dt.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''
    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([2, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in

    Y = _np.zeros([2, x_in.size])
    Y[0,:] = _np.ones(x_in.size)
    Y[1,:] = dx_in

    y = sigmoid(theta[:2].dot(X)) * sigmoid(theta[2:].dot(Y))

    return y

def SCI_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SCI model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            X = _np.array([1, VV[i,j], 0, 0])
            Y = _np.array([0, 0, 1, dVV[i,j]])
            resp[i,j] = sigmoid(theta.dot(X))*sigmoid(theta.dot(Y))

    return resp



'''
    Spiking Continuous Independent(SCI) objective function, gradient, gain
'''
def SCI_fobj(theta, x_in, y):
    '''
    Objective function and its gradient
    Likelihood objective function and its gradient of Spiking Continuous(SCI) spiking model.
    Returns the cost value(J) and the gradient(grad).
    see SCI(x_in, theta).

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
    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([theta.size, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in

    Y = _np.zeros([theta.size, x_in.size])
    Y[2,:] = _np.ones(x_in.size)
    Y[3,:] = dx_in

    y_est = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y))

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
    wx = _np.ones(y_est.shape) - sigmoid(theta.dot(X))
    wy = _np.ones(y_est.shape) - sigmoid(theta.dot(Y))
    grad = _np.sum(e * (wx*X + wy*Y),1)

    return J, grad

def SCI_gain(theta, x_in):
    '''
    Gain of the Spiking Continuous Independent(SCI) model.
    Compute the gain of the SCI spiking block model, which is the change in output at given the change in input(and its derivative).

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    gain (ndarray):
        The gain of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([theta.size, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in

    Y = _np.zeros([theta.size, x_in.size])
    Y[2,:] = _np.ones(x_in.size)
    Y[3,:] = dx_in

    y_est = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y))
    wx = _np.ones(y_est.shape) - sigmoid(theta.dot(X))
    # wy = _np.ones(y_est.shape) - sigmoid(theta.dot(Y))

    # cg = (49/20) / 0.001
    # gain = y_est * ( wx*theta[1] + wy*cg*theta[3] )
    gain = y_est * wx*theta[1]

    return gain

def SCI_gain1(theta, x, dx):
    '''
    returns SCI model gain at one point, (x, dx) given theta.
    gain(x, dx; theta)
    '''
    X = _np.array([1, x, 0, 0])
    Y = _np.array([0, 0, 1, dx])
    y_est = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y))
    wx = _np.ones(y_est.shape) - sigmoid(theta.dot(X))
    # wy = _np.ones(y_est.shape) - sigmoid(theta.dot(Y))

    # cg = (49/20) / 0.001
    # gain = y_est * ( wx*theta[1] + wy*cg*theta[3] )
    gain = y_est * wx*theta[1]

    return gain

def SCI_gain2d(theta, X, Y):
    '''
    returns 2 dimensional array SCI model 2 dimensonal gain  given meshgrid X, Y, and theta.
    gain(X, Y; theta)
    '''
    gain = _np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            gain[i,j] = SCI_gain1(theta, X[i,j], Y[i,j])

    return gain



'''
    Spiking Continuous Independent Feedback(SCIF)
'''
def SCIF(theta, x_in):
    '''
    Spiking Continuous Independent Feedback original code.
    Compute basic continuous 2D nonlinearity spiking block, with negative feedback.
    The 2D nonlinearity is a function of x and dx/dt.
    The negative feedback is an exponential function.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.

    h (ndarray):
        The internal variable h, which is the membrane potential added with feedback

    gain (ndarray):
        The instantaneous gain
    '''


    y = SCIF_C(theta, x_in)

    dx_in = deriv(x_in, 0.001)
    h = _st.SCIF_get_h(theta, x_in, dx_in)
    gain = _st.SCIF_gain(theta, x_in, dx_in)

    return y, h, gain



def SCIF_C(theta, x_in):
    '''
    Spiking Continuous Independent Feedback (C code).
    Compute fast basic continuous 2D nonlinearity spiking block, with negative feedback.
    The 2D nonlinearity is a function of x and dx/dt.
    The negative feedback is an exponential function.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''
    dx_in = deriv(x_in, 0.001)

    y = _st.SCIF(theta, x_in, dx_in)

    return y

def SCIF_constFB(theta, x_in):
    '''
    Spiking Continuous Independent Feedback with const Feedback(no effect of Feedback)
    '''

    dx_in = deriv(x_in, 0.001)
    h = _st.SCIF_get_h(theta, x_in, dx_in)
    b = h - x_in
    mean_b = _np.mean(b)

    h_const = x_in + mean_b

    y = SCI(theta[:4], h_const)
    gain = SCI_gain(theta[:4], h_const)

    return y, gain


def SCIF_fobj(theta, x_in, y):
    '''
    Spiking Continuous Independent Feedback(SCIF) objective
    '''
    J, grad = _obj.fobjective_numel(_obj.log_diff_fobj, SCIF_C, theta, (x_in, y))
    # J, grad = _obj.fobjective_numel(_obj.log_fobj, SCIF_C, theta, (x_in, y))

    return J, grad

def SCIF_bnds():
    '''
    return SCIF bound constraints.
    '''
    bnds = ((None,None),(0,None),(None,None),(0,None),(0,None),(0,None))

    return bnds


'''
    Spiking Continuous Independent Feedback with 2 feedback filters(SCIF2)
'''
def SCIF2(theta, x_in):
    '''
    Compute basic continuous 2D nonlinearity spiking block, with negative feedback.
    The 2D nonlinearity is a function of x and dx/dt.
    The negative feedback is an exponential function.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.

    h (ndarray):
        The internal variable h, which is the membrane potential added with feedback

    gain (ndarray):
        The instantaneous gain
    '''


    y = SCIF2_C(theta, x_in)

    dx_in = deriv(x_in, 0.001)
    h = _st.SCIF2_get_h(theta, x_in, dx_in)
    gain = _st.SCIF2_gain(theta, x_in, dx_in)

    return y, h, gain



def SCIF2_C(theta, x_in):
    '''
    Spiking Continuous Independent Feedback (C code).
    Compute fast basic continuous 2D nonlinearity spiking block, with negative feedback.
    The 2D nonlinearity is a function of x and dx/dt.
    The negative feedback is an exponential function.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''
    dx_in = deriv(x_in, 0.001)

    y = _st.SCIF2(theta, x_in, dx_in)

    return y

def SCIF2_constFB(theta, x_in):
    '''
    Spiking Continuous Independent Feedback with const Feedback(no effect of Feedback)
    '''

    dx_in = deriv(x_in, 0.001)
    h = _st.SCIF_get_h(theta, x_in, dx_in)
    b = h - x_in
    mean_b = _np.mean(b)

    h_const = x_in + mean_b

    y = SCI(theta[:4], h_const)
    gain = SCI_gain(theta[:4], h_const)

    return y, gain


def SCIF2_fobj(theta, x_in, y):
    '''
    Spiking Continuous Independent Feedback(SCIF) objective
    '''
    J, grad = _obj.fobjective_numel(_obj.log_diff_fobj, SCIF2_C, theta, (x_in, y))
    # J, grad = _obj.fobjective_numel(_obj.log_fobj, SCIF_C, theta, (x_in, y))

    return J, grad

def SCIF2_bnds():
    '''
    return SCIF bound constraints.
    '''
    bnds = ((None,None),(0,None),(None,None),(0,None),(0,None),(0,None),(0,None),(0,None))

    return bnds

'''
    Spiking Continuous Independent Ellipsoid(SCIE)
'''
def SCIE(theta, x_in):
    '''
    Spiking Continuous Independent Ellipsoid(SCIE) spiking model
    Returns the output firing rate, which is a product of sigmoid functions.

    Input
    -----
    theta (ndarray):
        parameters
    x_in (ndarray):
        input to the model (memebrane potential)

    Output
    ------
    y (ndarray):
        output of the model(firing rate)
    '''

    dx_in = deriv(x_in, 0.001)

    X = _np.zeros([2, x_in.size])
    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in

    Y = _np.zeros([2, x_in.size])
    Y[0,:] = _np.ones(x_in.size)
    Y[1,:] = dx_in

    Z = _np.zeros([6, x_in.size])
    Z[0,:] = _np.ones(x_in.size)
    Z[1,:] = x_in
    Z[2,:] = dx_in
    Z[3,:] = x_in * dx_in
    Z[4,:] = x_in ** 2
    Z[5,:] = dx_in ** 2


    y = sigmoid(theta[:2].dot(X)) * sigmoid(theta[2:4].dot(Y)) * sigmoid(theta[4:].dot(Z))

    return y

def SCIE_fobj(theta, x_in, y):
    '''
    Objective function and its gradient
    Likelihood objective function and its gradient of Spiking Continuous Independent Ellipsoid(SCIE) spiking model.
    Returns the cost value(J) and the gradient(grad).
    see SCI(x_in, theta).

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
    dx_in = deriv(x_in, 0.001)

    theta_size = theta.size
    X = _np.zeros([theta_size, x_in.size])
    Y = _np.zeros([theta_size, x_in.size])
    Z = _np.zeros([theta_size, x_in.size])

    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in

    Y[2,:] = _np.ones(x_in.size)
    Y[3,:] = dx_in

    Z[4,:] = _np.ones(x_in.size)
    Z[5,:] = x_in
    Z[6,:] = dx_in
    Z[7,:] = x_in * dx_in
    Z[8,:] = x_in ** 2
    Z[9,:] = dx_in ** 2

    y_est = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y)) * sigmoid(theta.dot(Z))

    '''
    likelihood objective function
    '''
    # temp = _np.log(y_est)
    # temp[_np.isinf(temp)] = -1e-6
    # J = _np.sum(y_est - y*temp)

    '''
    weighted objective function
    '''
    len_section = 10000
    num_section = _np.int(_np.floor(x_in.size / len_section))
    J = _np.zeros(num_section)
    for i in range(num_section):
        x_range = _np.arange(i*len_section, (i+1)*len_section)
        '''
        likelihood objective function
        '''
        temp = _np.log(y_est[x_range])
        temp[_np.isinf(temp)] = -1e-6
        weight = 1 / (_np.mean(y[x_range]) + 1e-3)
        J[i] = _np.sum(y_est[x_range] - y[x_range]*temp) * weight


    '''
    gradient of the objective function
    '''
    e = y_est - y
    wx = _np.ones(y_est.shape) - sigmoid(theta.dot(X))
    wy = _np.ones(y_est.shape) - sigmoid(theta.dot(Y))
    wz = _np.ones(y_est.shape) - sigmoid(theta.dot(Z))
    grad = _np.sum(e * (wx*X + wy*Y + wz*Z),1)

    return J, grad


def SCIE_1d(theta, x, dx):
    '''
    returns 2 dimensional array of SCIE model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''

    X = _np.array([1, x, 0, 0, 0, 0, 0, 0, 0, 0])
    Y = _np.array([0, 0, 1, dx, 0, 0, 0, 0, 0, 0])
    Z = _np.array([0, 0, 0, 0, 1, x, dx, x*dx, x**2, dx**2])

    resp = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y)) * sigmoid(theta.dot(Z))

    return resp


def SCIE_2d(theta, VV, dVV):
    '''
    returns 2 dimensional array of SCIE model 2 dimensonal response given meshgrid VV, dVV, and theta.
    resp(VV, dVV; theta)
    '''
    resp = _np.zeros(VV.shape)
    for i in range(VV.shape[0]):
        for j in range(VV.shape[1]):
            x = VV[i,j]
            dx = dVV[i,j]

            X = _np.array([1, x, 0, 0, 0, 0, 0, 0, 0, 0])
            Y = _np.array([0, 0, 1, dx, 0, 0, 0, 0, 0, 0])
            Z = _np.array([0, 0, 0, 0, 1, x, dx, x*dx, x**2, dx**2])

            resp[i,j] = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y)) * sigmoid(theta.dot(Z))

    return resp

def SCIE_gain(theta, x_in):
    '''
    Gain of the Spiking Continuous Independent Ellipsoid(SCIE) model.
    Compute the gain of the SCIE spiking block model, which is the change in output at given the change in input(and its derivative).

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    gain (ndarray):
        The gain of the spiking block.
    '''
    dx_in = deriv(x_in, 0.001)

    theta_size = theta.size
    X = _np.zeros([theta_size, x_in.size])
    Y = _np.zeros([theta_size, x_in.size])
    Z = _np.zeros([theta_size, x_in.size])

    X[0,:] = _np.ones(x_in.size)
    X[1,:] = x_in

    Y[2,:] = _np.ones(x_in.size)
    Y[3,:] = dx_in

    Z[4,:] = _np.ones(x_in.size)
    Z[5,:] = x_in
    Z[6,:] = dx_in
    Z[7,:] = x_in * dx_in
    Z[8,:] = x_in ** 2
    Z[9,:] = dx_in ** 2

    y_est = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y)) * sigmoid(theta.dot(Z))

    wx = _np.ones(y_est.shape) - sigmoid(theta.dot(X))
    wz = _np.ones(y_est.shape) - sigmoid(theta.dot(Z))

    gain = y_est * (theta[1] * wx + wz * (theta[5]*_np.ones(y_est.shape) + theta[7]*dx_in + 2*theta[8]*x_in))

    return gain


def SCIE_gain1(theta, x, dx):
    '''
    returns SCIE model gain at one point, (x, dx) given theta.
    gain(x, dx; theta)
    '''
    X = _np.array([1, x, 0, 0, 0, 0, 0, 0, 0, 0])
    Y = _np.array([0, 0, 1, dx, 0, 0, 0, 0, 0, 0])
    Z = _np.array([0, 0, 0, 0, 1, x, dx, x*dx, x**2, dx**2])

    y_est = sigmoid(theta.dot(X)) * sigmoid(theta.dot(Y)) * sigmoid(theta.dot(Z))

    wx = _np.ones(y_est.shape) - sigmoid(theta.dot(X))
    wz = _np.ones(y_est.shape) - sigmoid(theta.dot(Z))

    gain = y_est * (theta[1] * wx + wz * (theta[5]*_np.ones(y_est.shape) + theta[7]*dx + 2*theta[8]*x))

    return gain

def SCIE_gain2d(theta, X, Y):
    '''
    returns 2 dimensional array SCIE model 2 dimensonal gain  given meshgrid X, Y, and theta.
    gain(X, Y; theta)
    '''
    gain = _np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            gain[i,j] = SCIE_gain1(theta, X[i,j], Y[i,j])

    return gain


def SCIE_bnds():
    '''
    return SCIE bound constraints.
    '''
    bnds = ((None,None),(0,None),(None,None),(0,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None))

    return bnds



'''
    Spiking Continuous Independent Ellipsoid Feedback(SCIEF)
'''
def SCIEF(theta, x_in):
    '''
    Spiking Continuous Independent Ellipsoid Feedback.
    Compute basic continuous 2D nonlinearity spiking block, with negative feedback.
    The 2D nonlinearity is a function of x and dx/dt.
    The negative feedback is an exponential function.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.

    h (ndarray):
        The internal variable h, which is the membrane potential added with feedback

    gain (ndarray):
        The instantaneous gain
    '''


    dx_in = deriv(x_in, 0.001)
    y = _st.SCIEF(theta, x_in, dx_in)
    h = _st.SCIEF_get_h(theta, x_in, dx_in)
    b = h - x_in
    gain = _st.SCIEF_gain(theta, x_in, dx_in)

    theta_fb = theta[10:]
    len_fb = 5000
    t = _np.arange(len_fb)
    fb1 = theta_fb[0] * _np.exp( -t / theta_fb[1])
    fb2 = theta_fb[2] * _np.exp( -t / theta_fb[3])
    fb = (fb1 + fb2)/2

    return y, h, gain, fb, b



def SCIEF_C(theta, x_in):
    '''
    Spiking Continuous Independent Feedback (C code).
    Compute fast basic continuous 2D nonlinearity spiking block, with negative feedback.
    The 2D nonlinearity is a function of x and dx/dt.
    The negative feedback is an exponential function.

    Input
    -----
    x_in (ndarray):
        The input to the spiking block or subthreshold membrane potential.

    theta (ndarray):
        The SC parameters.

    Output
    ------
    y (ndarray):
        The firing rate, output of the spiking block.
    '''

    dx_in = deriv(x_in, 0.001)
    y = _st.SCIEF(theta, x_in, dx_in)

    return y


def SCIEF_fobj(theta, x_in, y):
    '''
    Spiking Continuous Independent Feedback(SCIF) objective
    '''
    J, grad = _obj.fobjective_numel(_obj.log_diff_fobj, SCIEF_C, theta, (x_in, y))
    # J, grad = _obj.fobjective_numel(_obj.log_fobj, SCIEF_C, theta, (x_in, y))

    return J, grad

def SCIEF_bnds():
    '''
    return SCIF bound constraints.
    '''
    # bnds = ((None,None),(0,None),(None,None),(0,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(0,None),(0,None))
    bnds = ((None,None),(0,None),(None,None),(0,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None),(0,None),(0,None),(0,None),(0,None))

    return bnds

def SCIEF_constFB(theta, x_in):
    '''
    Spiking Continuous Independent Feedback with const Feedback(no effect of Feedback)
    '''

    dx_in = deriv(x_in, 0.001)
    h = _st.SCIEF_get_h(theta, x_in, dx_in)
    b = h - x_in
    mean_b = _np.mean(b)

    h_const = x_in + mean_b

    y = SCIE(theta[:10], h_const)
    gain = SCIE_gain(theta[:10], h_const)

    return y, gain
'''
   Additional functions
'''
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



def deriv(x, dt):
    '''
    Return derivative of the input, using center finite derivative.

    Input
    -----
    x (ndarray):
        Input signal

    dt (double):
        delta t

    Output
    ------
    y (ndarray):
        Derivative of x
    '''

    filt = _np.array([-1, 8, 0, -8, 1]) / (12*dt)
    y = _np.convolve(x, filt)
    y = y[2:]
    y = y[:x.size]
    y[-2:] = y[-3]

    return y

def deriv_backward(x, dt):
    '''
    Returns the derivative of the input x using backward finite derivative.

    Input
    -----
    x (ndarray):
        Input signal

    dt (double):
        delta t

    Output
    ------
    y (ndarray):
        Derivative of x
    '''

    # backward derivative coefficients
    c = _np.array([49/20, -6, 15/2, -20/3, 15/4, -6/5, 1/6]) / dt
    X = _tpltz([x[0],0,0,0,0,0,0], x)

    dx = c.dot(X)

    return dx


def hist2d(X, Y, x, y, z):
    '''
    returns Average firing rate in 2-dimensions in x and y given meshgrid X, and Y.

    Input
    -----

    Output
    ------
    fr:
        average firing rate array (2-dimension, in x and y)
    '''

    hist = _np.zeros((X.shape[0]-1, X.shape[1]-1))
    temp = _np.ones(x.shape)

    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            # print(X[i,j], Y[i,j])
            idx_x = (x > X[i,j]) * (x < X[i,j+1])
            idx_y = (y > Y[i,j]) * (y < Y[i+1,j])
            idx_sum = idx_x * idx_y
            hist[i,j] = _np.sum(temp[idx_sum])

    hist[_np.isnan(hist)] = 0

    return hist

def get_averagefiring2d(X, Y, x, y, z):
    '''
    returns Average firing rate in 2-dimensions in x and y given meshgrid X, and Y.

    Input
    -----

    Output
    ------
    fr:
        average firing rate array (2-dimension, in x and y)
    '''

    fr = _np.zeros((X.shape[0]-1, X.shape[1]-1))

    for i in range(X.shape[0]-1):
        for j in range(X.shape[1]-1):
            idx_x = (x > X[i,j]) * (x < X[i,j+1])
            idx_y = (y > Y[i,j]) * (y < Y[i+1,j])

            fr[i,j] = _np.mean(z[(idx_x * idx_y)])

    fr[_np.isnan(fr)] = 0

    return fr


def gain2d(fgain, theta, X, Y, x, y, z):
    '''
    returns model firing rate in 2-dimensions in x and y given 1-dimensional gain function fgain.
    if x_lim and y_lim are given, then the model firing rate are computed in those range and return 0 for those out of range

    Input
    -----
    fgain (function):
        1-dimensional gain function

    theta (ndarray):
        the model parameter

    X (ndarray):
        the meshgrid of x

    Y (ndarray):
        the meshgrid of y

    x (ndarray):
        input x of the 2-dimensional nonlinearity

    y (ndarray):
        input y of the 2-dimensional nonlinearity

    z (ndarray):
        output z of the 2-dimensional nonlinearity

    Output
    ------
    gain (ndarray):
        the 2-dimensional gain
    '''

    hist_2d = hist2d(X, Y, x, y, z)

    gain = _np.zeros(X.shape)

    for i in range(X.shape[0]-1):
        y_lim = (hist_2d[i,:]>0)
        y_range = _np.arange(len(y_lim))
        y_range = y_range[y_lim]
        # for j in range(X.shape[1]):
        if (y_range.size):
            for j in y_range:
                gain[i,j] = fgain(theta, X[i,j], Y[i,j])

    gain[_np.isnan(gain)] = 0

    return gain

def resp2d(f, theta, X, Y, x, y, z):

    hist_2d = hist2d(X, Y, x, y, z)

    resp = _np.zeros(X.shape)

    for i in range(X.shape[0]-1):
        y_lim = (hist_2d[i,:]>0)
        y_range = _np.arange(len(y_lim))
        y_range = y_range[y_lim]
        # for j in range(X.shape[1]):
        if (y_range.size):
            for j in y_range:
                resp[i,j] = f(theta, X[i,j], Y[i,j])

    resp[_np.isnan(resp)] = 0

    return resp





'''
Spiking Discrete Feedback Model
    Spiking threshold and feedback part of the Keat's model.
    Implementation: SDF
'''
def SDF(theta, x_in, gwin_std=10, gwin_len=1000):
    '''
    Spiking Discrete Feedback (C code)

    Input
    -----
    theta (ndarray):
        input parameter to the model

    x_in (ndarray):
        input membrane potential(input signal) to the model

    Output
    ------
    r (ndarray):
        output firing rate response
    '''

    spikes = _st.SDF(theta, x_in)
    r = _dpt.Spk2FR_1(spikes.T, gwin_len, gwin_std)

    return r

def SDF_C(theta, x_in):
    '''
    Spiking Discrete Feedback (C code)

    Input
    -----
    theta (ndarray):
        input parameter to the model

    x_in (ndarray):
        input membrane potential(input signal) to the model

    Output
    ------
    r (ndarray):
        output response(spikes: 1's and 0's), 1's indicate spikes.
    '''

    r = _st.SDF(theta, x_in)

    return r



def SDF_orig(theta, x_in):
    '''
    Spiking Discrete Feedback

    Input
    -----
    theta (ndarray):
        input parameter to the model

    x_in (ndarray):
        input membrane potential(input signal) to the model

    Output
    ------
    r (ndarray):
        output response(spikes: 1's and 0's), 1's indicate spikes.
    '''

    # output response
    N = x_in.size
    r = _np.zeros(N)
    g = _np.zeros(N)

    # extract parameters
    thr = theta[0]
    amp1 = theta[1]
    tau1 = theta[2]
    amp2 = theta[3]
    tau2 = theta[4]

    # feedback filter
    len_filt_fb = 1000
    t_fb = _np.arange(len_filt_fb)
    filt_fb = (-amp1 * _np.exp(-t_fb / tau1)) + (-amp2 * _np.exp(-t_fb / tau2))
    filt_fb /= 2

    # internal variable
    g[0] = x_in[0]
    for i in range(1,N):
        g[i] += x_in[i]
        if (g[i] > thr) and (g[i] > g[i-1]):
            r[i] = 1

            if (i < N - len_filt_fb):
                # print(len(g[i+1:i+1+len_filt_fb]), len(filt_fb))
                g[i+1:i+1+len_filt_fb] += filt_fb

            else:
                if not (i == N-1):
                    g[i+1:N] += filt_fb[:len(range(i+1,N))]
    return r, g

def SDF_fobj(theta, x_in, y):
    '''
    Spiking Continuous Independent Feedback(SCIF) objective
    '''
    J, grad = _obj.fobjective_numel(_obj.log_diff_fobj, SDF, theta, (x_in, y))

    return J, grad

def SDF_bnds():
    '''
    return SCIF bound constraints.
    '''
    bnds = ((None,None),(0,None),(0,None),(0,None),(0,None))

    return bnds

import time
def main():
    x = _np.sin(_np.linspace(0, 300*_np.pi, 300000))
    # theta = _np.array([-1,1,0,0.1])
    # temp = [1,1,0,0,1,100]
    # theta = _np.array(temp)
    # theta = _np.sin(_np.linspace(0, _np.pi, 6))
    theta = _np.zeros(6)
    theta[0] = -1
    theta[1] = 1
    theta[2] = -0.5
    theta[3] = 0.1
    theta[4] = 0.1
    theta[5] = 100
    # y = SC(theta, x)
    y1 = SCI(theta[:4], x)
    start_time = time.time()
    y = SCIF_C(theta, x)
    print(len(y1), len(y))
    print("--- %s seconds ---" % str(time.time() - start_time))

if __name__ == '__main__':
    main()
