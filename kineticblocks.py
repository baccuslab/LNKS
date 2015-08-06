'''
kineticblocks.py

Implementation of different Kinetics Block topologies

author: Bongsoo Suh
created: 2015-01-27

(C) 2015 bongsoos
'''

import numpy as _np
import kinetictools as _kt

def K4S(p, Xinit, u):
    '''
    The Kinetics block operation, original slow python code.

    Input
    -----
    p (ndarray):
        The kinetics parameters of the LNK model.

    Xinit (ndarray):
        The initial states of the Kinetics block.

    u (ndarray):
        The input signal to the Kinetics block.

    Output
    ------
    X (ndarray):
        The responses of the four states(R, A, I1, I2).
    '''
    dt = 0.001# * _np.ones(4)
    X = _np.zeros((4,u.size))
    X[:,0] = Xinit

    for t in range(u.size-1):
        Qt = _np.array([ [-(p[0]*u[t]+p[6]), 0, p[2], 0],
                        [(p[0]*u[t]+p[6]), -p[1], 0, 0],
                        [0, p[1],-(p[2]+p[3]), p[4]*u[t]+p[5]],
                        [0, 0, p[3], -(p[4]*u[t]+p[5])] ])
        X[:,t+1] = X[:,t] + dt * Qt.dot(X[:,t]) 

    return X


def K4S_C(p, Xinit, u):
    '''
    The Fast Kinetics block operation, calling computeKinetics.c code.

    Input
    -----
    p (ndarray):
        The kinetics parameters of the LNK model.

    Xinit (ndarray):
        The initial states of the Kinetics block.

    u (ndarray):
        The input signal to the Kinetics block.

    Output
    ------
    X (ndarray):
        The responses of the four states(R, A, I1, I2).

    Variables
    ---------
    dt (double):
        sampling time/period

    X (ndarray):
        The responses of the four states(R, A, I1, I2).
    '''

    dt = 0.001
    X = _np.zeros((Xinit.size, u.size))
    X[:,0] = Xinit
    X = _kt.K4S(p, X, u, int(Xinit.size), int(u.size), dt) # import kinetictools

    return X
