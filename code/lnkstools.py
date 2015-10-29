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
import optimizationtools as _ot


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
    f, g, u, thetaK, X, v = LNK(theta[:17], stim, pathway=1)
    #r = Spiking(v, theta[16:])
    r = _sb.SCIE(theta[17:], v)

    return f, g, u, thetaK, X, v, r

def LNKS_f(theta, stim, pathway=1):
    '''
    compute LNKS model for optimization
    '''
    f, g, u, thetaK, X, v, r = LNKS(theta, stim, pathway=pathway)

    return r


def LNKS_fobj(theta, stim, y):
    '''
    LNKS model objective function
    '''
    #J, grad = _ot.fobjective_numel(LNK_fobj_helper, LNK_f, theta, (stim, y))

    #return J, grad
    J = LNKS_fobj_helper(LNKS_f, theta, stim, y)

    return J


def LNKS_fobj_helper(LNKS_f, theta, stim, y):
    '''
    LNKS model objective function helper function

    Weighted sum of log-likelihood and mean-square error
    '''

    J1 = _ot.log_fobj(LNKS_f, theta, stim, y)
    J2 = _ot.diff_fobj(LNKS_f, theta, stim, y)
    J = J1 + J2

    return J

def LNKS_bnds(theta=None, bnd_mode=0):
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
        bnds = None

    elif bnd_mode == 1:
        bnd_LNK = tuple([(None, None) for i in range(17)])
        bnd_S = tuple([(theta[i],theta[i]) for i in range(17,theta.size)])
        bnd = bnd_LNK + bnd_S

    elif bnd_mode == 2:
        bnd_LNK = tuple([(theta[i],theta[i]) for i in range(17)])
        bnd_S = ((None,None),(0,None),(None,None),(0,None),(None,None),(None,None),(None,None),(None,None),(None,None),(None,None))
        bnd = bnd_LNK + bnd_S


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


    # two pathway
    elif pathway == 2:
        print("two pathway")
    else:
        print("not available")


    return f, g, u, thetaK, X, v

def LNK_f(theta, stim, pathway=1):
    '''
    compute LNK model for optimization
    '''
    f, g, u, thetaK, X, v = LNK(theta, stim, pathway=pathway)

    return v


def LNK_fobj(theta, stim, y):
    '''
    LNK model objective function
    '''
    #J, grad = _ot.fobjective_numel(LNK_fobj_helper, LNK_f, theta, (stim, y))

    #return J, grad
    J = LNK_fobj_helper(LNK_f, theta, stim, y)

    return J


def LNK_fobj_helper(LNK_f, theta, stim, y):
    '''
    LNK model objective function helper function

    Weighted sum of mean-square error
    '''

    v = LNK_f(theta, stim)

    len_section = 10000
    num_sections = _np.int(_np.floor(stim.size / len_section))
    J_sections = _np.zeros(num_sections)

    for i in range(num_sections):
        if i == 0:
            range_section = _np.arange(1000,len_section*(i+1))
        else:
            range_section = _np.arange(len_section*i,len_section*(i+1))
        diff = y[range_section] - v[range_section]
        y_std = _np.std(y[range_section])
        if y_std == 0:
            y_std = 1e-6
        J_sections[i] = _np.sum((diff**2) / (y_std**2))

    J = _np.sum(J_sections)

    return J

def LNK_bnds():

    bnds = None

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
    #X = _kb.K4S(p, Xinit, xinput)

    # Fast Kinetics block computation
    X =  _kb.K4S_C(p, Xinit, xinput)

    return X


def Spiking(theta, x_in):
    '''
    Compute firing rate transformation from membrane potential

    Input
    -----

    Output
    ------
    '''

    y, h, gain = SCIF(theta, x_in)

    return y, h, gain


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

