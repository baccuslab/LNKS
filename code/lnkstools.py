#!/usr/bin/env python3

'''
lnkstools.py

Tools for computing and optimzing LNKS, LNK, Spiking models.

author: Bongsoo Suh
created: 2015-01-27

(C) 2015 bongsoos
'''

import numpy as _np
from scipy.linalg import orth as _orth
import kineticblocks as _kb
import spikingblocks as _sb
import objectivetools as _obj


def LNKS(theta, stim, pathway=1):
    '''
    Compute LNKS model output of a cell given input stimulus
    Input
    -----
    theta (ndarray):
        The LNKS parameters.

    stim (ndarray):
        The stimulus array.

    pathway (int):
        Number of pathways. 1 or 2 available. Otherwise error.


    Output
    ------
    f (ndarray):
        The linear filter of the LNK model.

    g (ndarray):
        The output of the linear filter of the LNK model.

    u (ndarray):
        The output of the nonlinearity of the LNK model.

    K (ndarray):
        The kinetics parameters of the LNK model.

    X (ndarray):
        The responses of four states(R, A, I1, I2).

    v (ndarray):
        The LNK model output(membrane potential).

    r (ndarray):
        The LNKS model output.

    '''

    if pathway == 1:
        # compute LNK model
        f, g, u, thetaK, X, v = LNK(theta[:17], stim, pathway)

        # Comptue Spiking model
        # Running fast c-extension
        # To get the internal variables of the spiking block, use Spiking method.
        r = _sb.SC1DF_C(theta[17:], v)

    elif pathway == 2:
        # compute LNK model
        # One path LNK parameter 17 plus 1 weight, thus 18
        # Two path LNK parameter (17*2 + 2 = 36) where two weights are w_on and w_off
        # theta[17] and theta[35] respectively.
        f, g, u, thetaK, X, v = LNK(theta[:36], stim, pathway)

        # Comptue Spiking model
        # spiking parameters following after LNK parameters
        r = _sb.SC1DF_C(theta[36:], v)

    else:
        raise ValueError('The pathway parameter should be 1 or 2')


    return f, g, u, thetaK, X, v, r

def LNKS_f(theta, stim, pathway=1):
    '''
    compute LNKS model for optimization using only firing rate as output
    '''
    f, g, u, thetaK, X, v, r = LNKS(theta, stim, pathway)

    return r


def LNKS_fobj(theta, stim, y, pathway=1):
    '''
    LNKS model objective function for using only firing rate as output
    Returns objective value(J) and gradient(grad)

    Inputs
    ------
        theta: model parameters
        stim: input data
        y: output data (fr)
        pathway (int): LNK pathway (1 or 2)

    Outputs
    -------
        J: objective value
        grad: gradient of objective
    '''

    J = LNKS_fobj_helper(LNKS_f, theta, stim, y, pathway)
    grad = _obj.fobj_numel_grad(LNKS_fobj_helper, LNKS_f, theta, stim, y, pathway)

    return J, grad


def LNKS_fobj_helper(f, theta, stim, y, pathway=1):
    '''
    LNKS model objective function helper function

    Weighted sum of log-likelihood and mean-square error
    '''

    y_est = f(theta, stim, pathway)

    # linear combination of objective functions
    J_poss = _obj.poisson_weighted_loss(y, y_est, len_section=10000, weight_type="mean")
    J_mse = _obj.mse_weighted_loss(y, y_est, len_section=10000, weight_type="mean")
    J = J_poss + J_mse

    return J


def LNKS_MP_f(theta, stim, pathway=1):
    '''
    compute LNKS model for optimization using both membrane potential and firing rate
    '''
    f, g, u, thetaK, X, v, r = LNKS(theta, stim, pathway)

    return v, r


def LNKS_MP_fobj(theta, stim, y_data, pathway=1):
    '''
    LNKS model objective function for using both membrane potential and firing rate
    Returns objective value(J) and gradient(grad)

    Inputs
    ------
        theta: model parameters
        stim: input data
        y_data: output data tuple (mp, fr)
        pathway (int): LNK pathway (1 or 2)

    Outputs
    -------
        J: objective value
        grad: gradient of objective
    '''

    J = LNKS_MP_fobj_helper(LNKS_MP_f, theta, stim, y_data, pathway)
    grad = _obj.fobj_numel_grad(LNKS_MP_fobj_helper, LNKS_MP_f, theta, stim, y_data, pathway)

    return J, grad


def LNKS_MP_fobj_helper(f, theta, stim, y_data, pathway=1):
    '''
    LNKS model objective helper function for using both membrane potential and firing rate
    returns objective function value J

    Inputs
    -------
        f: LNKS model
        theta: model parameters
        stim: input data
        y_data: output data tuple (mp, fr)
        pathway (int): LNK pathway (1 or 2)

    Outputs
    -------
        J: objective value
    '''
    # data
    y_mp = y_data[0]
    y_fr = y_data[1]

    # model output
    y_mp_est, y_fr_est = f(theta, stim, pathway)

    # linear combination of objective functions
    J_mp = _obj.mse_weighted_loss(y_mp, y_mp_est, len_section=10000, weight_type="std")
    J_fr_poss = _obj.poisson_weighted_loss(y_fr, y_fr_est, len_section=10000, weight_type="mean")
    J_fr_mse = _obj.mse_weighted_loss(y_fr, y_fr_est, len_section=10000, weight_type="mean")
    J_fr = J_fr_poss + J_fr_mse

    J = J_mp + J_fr

    return J



def LNKS_bnds(theta=None, pathway=1, bnd_mode=0):
    '''
    LNKS parameter bounds for optimization

    Input
    -----
    theta (ndarray):
        initial LNKS parameters (theta)

    bnd_mode (int):
        Different modes of LNKS model parameter boundary for optimization
        0: fit LNKS model
        1: fit LNK (S fixed)
        2: fit S (LNK fixed)

        this is used with fitmodel.py (3x-optim/LNKS/program/)
    '''

    if bnd_mode == 0:
        bnd_S = _sb.SC1DF_bnds()
        bnd_LNK = LNK_bnds(pathway)
        bnds = bnd_LNK + bnd_S

    elif bnd_mode == 1:
        bnd_LNK = LNK_bnds(pathway)
        if pathway == 1:
            bnd_S = tuple([(theta[i],theta[i]) for i in range(17,theta.size)])
        elif pathway == 2:
            bnd_S = tuple([(theta[i],theta[i]) for i in range(36,theta.size)])
        else:
            raise ValueError('The pathway parameter should be 1 or 2')

        bnds = bnd_LNK + bnd_S

    elif bnd_mode == 2:
        if pathway == 1:
            bnd_LNK = tuple([(theta[i],theta[i]) for i in range(17)])
        elif pathway == 2:
            bnd_LNK = tuple([(theta[i],theta[i]) for i in range(36)])
        else:
            raise ValueError('The pathway parameter should be 1 or 2')

        bnd_S = _sb.SC1DF_bnds()
        bnds = bnd_LNK + bnd_S

    return bnds


def LNK(theta, stim, pathway=1):
    '''
    Compute LNK model output of a cell given input stimulus

    Input
    -----
    stim (ndarray):
        The stimulus array.

    theta (ndarray):
        The LNK parameters.

    pathway (int):
        Number of pathways. 1 or 2 available. Otherwise error.

    Output
    ------
    f (ndarray):
        The linear filter of the LNK model.

    g (ndarray):
        The output of the linear filter of the LNK model.

    u (ndarray):
        The output of the nonlinearity of the LNK model.

    K (ndarray):
        The kinetics parameters of the LNK model.

    X (ndarray):
        The responses of four states(R, A, I1, I2).

    v (ndarray):
        The LNK model output.

    '''
    basis = LinearFilterBasis_8param()
    nzstim = stim - _np.mean(stim) # nzstim: mean subtracted stimulus
    lenStim = nzstim.size
    numBasis = basis.shape[1] # number of Basis == number of L parameters

    # one pathway
    if pathway == 1:
        f, g, u, thetaK, X, v = LNK_single_path(theta, nzstim, lenStim, basis, numBasis)

    # two pathway
    elif pathway == 2:
        # first half: on pathway
        # second half: off pathway
        theta_on = theta[:17]
        theta_off = theta[18:35]

        # weights
        w_on = theta[17]
        w_off = theta[35]

        # compute single pathway LNK for each pathway
        f1, g1, u1, thetaK1, X1, v_on = LNK_single_path(theta_on, nzstim, lenStim, basis, numBasis)
        f2, g2, u2, thetaK2, X2, v_off = LNK_single_path(theta_off, nzstim, lenStim, basis, numBasis)

        # combine two path outputs
        f = [f1,f2]
        g = [g1,g2]
        u = [u1,u2]
        thetaK = [thetaK1,thetaK2]
        X = [X1,X2]

        # linear combination of on and off path outputs
        v = w_on * v_on + w_off * v_off

    else:
        raise ValueError('The pathway parameter should be 1 or 2')


    return f, g, u, thetaK, X, v


def LNK_single_path(theta, nzstim, lenStim, basis, numBasis):
    '''
    This method is a helper function for the LNK method, by computing a single path of a LNK model.
    LNK model could be a 1 path or a 2 path way model, where the single path method is same as the LNK model
    for case when path is 1, and single path is computed for twice for 2 path model and their outputs are summed.

    Input
    -----
    theta (ndarray):
        The parameters for single path.

    nzstim (ndarray):
        The mean subtracted stimulus array.

    lenStim (int):
        The length of the stimulus

    basis (ndarray):
        Linear filter basis

    numBasis (int):
        The number of basis

    Output
    ------
    f (ndarray):
        The linear filter of the LNK model.

    g (ndarray):
        The output of the linear filter of the LNK model.

    u (ndarray):
        The output of the nonlinearity of the LNK model.

    K (ndarray):
        The kinetics parameters of the LNK model.

    X (ndarray):
        The responses of four states(R, A, I1, I2).

    v (ndarray):
        The LNK model output.
    '''
    # Compute the linear filter and find the filtered output
    f = basis.dot(theta[:numBasis])
    g = _np.convolve(nzstim, f)
    g = g[:lenStim]

    # Compute nonlinearity, implemented by sigmoidal function
    u = sigmoid(theta[numBasis] + theta[numBasis+1] * g)

    # Compute Kinetics block operation.
    thetaK = _np.array(theta[numBasis+2:])
    X0 = _np.array([0.1,0.2,0.7,99]) # Initial Kinetics states
    X = Kinetics(thetaK, X0, u)
    v = X[1,:]

    v = v - _np.min(v)
    v = v / _np.max(v)

    return f, g, u, thetaK, X, v


def LNK_f(theta, stim, pathway=1):
    '''
    compute LNK model for optimization
    '''
    f, g, u, thetaK, X, v = LNK(theta, stim, pathway)

    return v


def LNK_fobj(theta, stim, y, pathway=1):
    '''
    LNK model objective function
    Returns objective value(J) and gradient(grad)

    Inputs
    ------
        theta: model parameters
        stim: input data
        y: output data (mp)
        pathway (int): LNK pathway (1 or 2)

    Outputs
    -------
        J: objective value
        grad: gradient of objective
    '''

    J = LNK_fobj_helper(LNK_f, theta, stim, y, pathway)
    grad = _obj.fobj_numel_grad(LNK_fobj_helper, LNK_f, theta, stim, y, pathway)

    return J, grad


def LNK_fobj_helper(LNK_f, theta, stim, y, pathway=1):
    '''
    LNK model objective function helper function

    Weighted sum of mean-square error
    '''

    v = LNK_f(theta, stim, pathway)

    J = _obj.mse_weighted_loss(y, v, len_section=10000, weight_type="std")

    return J


def LNK_bnds(pathway=1):
    '''
    Return boundaries for LNK model optimization
    '''

    if pathway == 1:
        bnds = L_bnds() + N_bnds() + K_bnds()

    elif pathway == 2:
        bnds = (L_bnds() + N_bnds() + K_bnds() + W_bnds()) * 2

    else:
        raise ValueError('The pathway parameter should be 1 or 2')

    return bnds

def L_bnds(numBasis=8):
    '''
    Return boundaries for L model
    '''
    bnds = tuple([(None,None) for i in range(numBasis)])

    return bnds

def N_bnds():
    '''
    Return boundaries for N model
    '''
    bnds = tuple([(None,None),(0,None)])

    return bnds

def K_bnds():
    '''
    Return boundaries for K model
    '''
    bnds = tuple([(0,None) for i in range(7)])

    return bnds

def W_bnds():
    '''
    Return boundaries for W weights for two pathway LNK model
    '''
    bnds = tuple([(0,1),(0,1)])

    return bnds


def Kinetics(p, Xinit, xinput):
    '''
    The Kinetics block operation wrapper function.

    Input
    -----
    p (ndarray):
        The kinetics parameters of the LNK model.

    Xinit (ndarray):
        The initial states of the Kinetics block.

    xinput (ndarray):
        The input signal to the Kinetics block(output of the LN of LNK, u(t)).

    Output
    ------
    X (ndarray):
        The responses of the four states(R, A, I1, I2).

    '''

    # Compute Kinetics block
    # X = _kb.K4S(p, Xinit, xinput)

    # Fast Kinetics block computation
    X = _kb.K4S_C(p, Xinit, xinput)

    return X


def Spiking(theta, x_in):
    '''
    Compute firing rate transformation from membrane potential

    Input
    -----

    Output
    ------
    '''

    y, h, gain, fb, b = _sb.SC1DF(theta, x_in)

    return y, h, gain, fb, b


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


import time
def main():
    stim = _np.random.rand(300000)
    theta = _np.random.rand(16)
    theta = _np.zeros(22)
    param = _np.array([0.0357,-0.3942,1.1002,-2.6024,2.8945,-2.6614,-3.4418,-0.0381,-5.4889,34.2319,5.0262,4.2102,0.8998,0.0022,0.0025,1.9985])
    theta[:16] = param
    theta[16] = -1
    theta[17] = 1
    theta[18] = 0
    theta[19] = 0.1
    theta[20] = 0
    theta[21] = 100

    start_time = time.time()
    f, g, u, K, X, v, r = LNKS(theta, stim)
    print("--- %s seconds ---" % str(time.time() - start_time))


if __name__ == '__main__':
    main()

