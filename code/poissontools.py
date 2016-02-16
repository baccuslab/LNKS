'''
poissontools.py

Tools for computing empirical probability distributions
and analyzing information transmission of Poisson noise model

Methods
-------
  PoissonNoiseModel
  PoissonNoiseModel_helper
  PoissonNoiseModel_helper2
  PoissonSpiking
  entropy
  entropy_conditional
  getSpikeCount
  mutualInfoPoisson
  probability_spiking
  hist2d

author: Bongsoo Suh
created: 2016-02-01

(C) 2016 bongsoos
'''

import numpy as _np
import pdb
from scipy.stats import poisson as _poisson
import dataprocesstools as _dpt
import infotools as _ift
import objectivetools as _obj
import lnkstools as _lnks
import optimizationtools as _ot
import analysistools_bak as _at

EPS = 0.0000000001

def probability_spiking(y):
    '''
    y (ndarray): spike count
    '''
    N_range = _np.array(range(_np.int_(_np.max(y))))
    y_, = _np.reshape(y, (1, y.shape[0]*y.shape[1]))
    P_y = _np.array([len(y[y==n]) for n in N_range])/len(y_)

    return P_y, N_range

def entropy(Px):
    '''
    compute entropy of x, H(x), given probability dist Px
    '''
    Px += EPS
    Hx = -_np.sum(_np.array(Px * _np.log2(Px)))

    return Hx

def entropy_conditional(P_y_x, Px, msc):
    '''
    compute conditional entropy of y given x, H(y|x), given conditional probability dist of y given x, P_y_x
    and given probability dist of x, Px
    '''
    Hyx = -_np.sum(P_y_x[:, list(msc)] * _np.log2(P_y_x[:, list(msc)]+EPS)) * Px

    return Hyx


def mutualInfoPoisson(fr, num_trials=10, bin_size=50, num_bins=10):
    '''
    compute mutual information between input stimulus(or filtered stimulus) x and output firing rate y
    using Poisson noise in bits and bits/spike(or bits/Hz)

    Inputs
    ------
    fr (ndarray)

    Output
    ------
        I (double):
            mutual information(bits)
        I_bps (double):
            mutual information(bits/spike)
    '''
    spikes, psth, rate = PoissonSpiking(fr, num_trials)
    spikeCounts = getSpikeCount(spikes, bin_size)

    P_spikeCounts, spikeCounts_range = probability_spiking(spikeCounts)
    P_noise, P_x, x_axis, y_axis, msc = PoissonNoiseModel(spikes, bin_size, num_bins)

    mean_X = _np.sum(x_axis * P_x) + EPS
    H_spikes = entropy(P_spikeCounts)
    H_spikes_x = entropy_conditional(P_noise, bin_size/len(fr), msc)

    I = H_spikes - H_spikes_x
    I_bps = I / mean_X
    return I, I_bps


def getSpikeCount(spikes, bin_size=10):
    '''
    return spike counts in each bin of bin_size for each trial
    spikes (ndarray)
    bin_size (int)
    '''
    spikeCount = _np.array([[_np.sum(spk[bin_size*i:bin_size*(i+1)]) for i in range(len(spk)//bin_size)] for spk in spikes ])

    return spikeCount

def PoissonSpiking(y, num_trials=10, dt=0.001, gwin_len=1000, gwin_std=10):
    '''
    Poisson spike generation
    Return spiking raster matrix given firing rate y

    Input
    -----
    y (ndarray)
    num_trials (int)
    dt (double): time interval (default 1ms)
    gwin_len (double)
    gwin_std (double)

    Output
    ------
    spikes (ndarray)
    psth (ndarray)
    rate (ndarray)

    Performs the function below:
        r = []
        U = _np.random.uniform(0,1,len(y))
        prob = _np.exp(-y*dt) * y*dt
        spikes = _np.int_(_np.array(U<prob))

    Output
    ------
    spikes (ndarray)
    '''

    _np.random.seed(77)
    spikes = _np.array([_np.int_(_np.array(_np.random.uniform(0,1,len(y)) < (1-_np.exp(-y*dt)))) for n in range(num_trials)])
    psth = _np.sum(spikes,0)/num_trials
    rate = _dpt.Spk2FR(spikes, gwin_len, gwin_std)/dt

    return spikes, psth, rate

def PoissonNoiseModel(spikes, bin_size=10, num_bins=10):
    '''
    This creates Poisson noise model(2 dimensional probability distribution),
    number of spikes(spike count) vs. firing rate(average rate on a bin)

    Input
    -----
    spikes (ndarray)
    bin_size
    num_bins

    Output
    ------
    P_noise (ndarray)
    '''

    num_trials = spikes.shape[0]
    spikeCount = getSpikeCount(spikes, bin_size)
    meanSpikeCount = _np.mean(spikeCount, 0)

    spikeCount_temp, = _np.reshape(_np.array([spikeCount[i,:] for i in range(num_trials)]),(1,num_trials*spikeCount.shape[1]))
    meanSpikeCount_temp, = _np.reshape(_np.array([meanSpikeCount for i in range(num_trials)]),(1,num_trials*spikeCount.shape[1]))

    P_noise, x_axis, y_axis = PoissonNoiseModel_helper(meanSpikeCount_temp, spikeCount_temp, num_bins)
    P_noise += EPS

    P_x = _np.sum(P_noise, 0) + EPS
    for i in range(P_noise.shape[1]):
        P_noise[:,i] = P_noise[:,i] / _np.sum(P_noise[:,i])

    msc = PoissonNoiseModel_helper2(meanSpikeCount, x_axis)

    return P_noise, P_x, x_axis, y_axis, msc

def PoissonNoiseModel_helper(x, y, num_bins=10, x_range=None, y_range=None):
    '''
    x (ndarray): frSampled
    y (ndarray): spikeCount
    '''

    n_x = num_bins+1

    if x_range:
        x_axis = _np.linspace(x_range[0], x_range[1], n_x)
    else:
        x_axis = _np.linspace(_np.min(x), _np.max(x), n_x)

    if x_range:
        y_axis = _np.arange(y_range)
    else:
        y_axis = _np.arange(_np.max(y))

    XX, YY = _np.meshgrid(x_axis, y_axis)
    # extent = [x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]]

    Hist = hist2d(XX, YY, x, y, 1)
    P_xy = Hist / _np.sum(Hist)

    x_axis = _np.array([(x_axis[i]+x_axis[i+1])/2 for i in range(len(x_axis)-1)])

    return P_xy, x_axis, y_axis

def PoissonNoiseModel_helper2(meanSpikeCount, x_axis):
    '''
    return mean spike count indices
    '''
    msc = _np.array([list(x_axis)] * len(meanSpikeCount)).T
    resp = _np.array([meanSpikeCount] * len(x_axis))
    # return x_axis[_np.argmin(_np.abs(resp - msc), 0)]
    return _np.argmin(_np.abs(resp - msc), 0)

def hist2d(X, Y, x, y, z):
    '''
    returns 2-dimensional histogram given x and y and their meshgrid X, and Y.

    Input
    -----
    X (ndarray): meshgrid
    Y (ndarray): meshgrid
    x (ndarray): input variable
    y (ndarray): output variable
    z (ndarray): 3rd dimensional variable dependent on both x and y,
                 such as firing rate, a function of membrane potential and its derivative

    Output
    ------
    hist (ndarray):
         2-dimension histogram given x and y
    '''

    hist = _np.zeros((X.shape[0], Y.shape[1]-1))
    temp = _np.ones(x.shape)

    for i in range(X.shape[0]):
        for j in range(Y.shape[1]-1):
            idx_x = (x >= X[i,j]) * (x <= X[i,j+1])
            idx_y = (y == Y[i,j])
            idx_sum = idx_x * idx_y
            hist[i,j] = _np.sum(temp[idx_sum])

    hist[_np.isnan(hist)] = 0

    return hist


def mutualInfo_2D(N_thrs, S_thrs, theta, stim):
    '''
    N_thrs (ndarray)
    S_thrs (ndarray)
    '''
    XX, YY = _np.meshgrid(N_thrs, S_thrs)
    MIs = _np.zeros([XX.shape[0], XX.shape[1]])
    MIs_bps = _np.zeros([XX.shape[0], XX.shape[1]])

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            theta[8] = - XX[i,j]
            theta[18] = - YY[i,j]
            f, g_temp, u, thetaK, X, v_est_temp, fr_est_temp = _lnks.LNKS(theta, stim, 1)
            MIs[i,j], MIs_bps[i,j] = mutualInfoPoisson(fr_est_temp)

    return MIs, MIs_bps

def mutualInfo_adaptiveIndex(N_thrs, S_thrs, theta, stim, contrast, fr):
    XX, YY = _np.meshgrid(N_thrs, S_thrs)
    MIs = _np.zeros([XX.shape[0], XX.shape[1]])
    MIs_bps = _np.zeros([XX.shape[0], XX.shape[1]])
    AIs = _np.zeros([XX.shape[0], XX.shape[1]])

    idx_contrast = _np.argsort(contrast)
    x_contrast = contrast[idx_contrast]
    SEC_LENGTH = 20000
    OFFSET = 5000

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            theta[8] = - XX[i,j]
            theta[18] = - YY[i,j]
            f, g_temp, u, thetaK, X, v_est_temp, fr_est_temp = _lnks.LNKS(theta, stim, 1)
            fr_est_temp = fr_est_temp * _np.max(fr)
            MIs[i,j], MIs_bps[i,j] = mutualInfoPoisson(fr_est_temp)

            gain_changes = _np.zeros(15)
            for k in range(len(idx_contrast)):
                stim_ = stim[k*SEC_LENGTH + OFFSET:(k+1)*SEC_LENGTH]
                fr_est_temp_ = fr_est_temp[k*SEC_LENGTH + OFFSET:(k+1)*SEC_LENGTH]
                LN_est_ln, L_est_ln, f_ln, theta_ln, LNresult = _at.LN_model(stim_, fr_est_temp_, num_bins=50)
                gain_changes[k] = theta_ln[1]

            c_avg, AI_avg = _at.get_avg_contrast(x_contrast, gain_changes[idx_contrast])
            line_est, line_theta = _ot.fit_line(c_avg[1:], AI_avg[1:]/AI_avg[-1])
            AIs[i,j] = line_theta[1]

    return MIs, MIs_bps, AIs


def mutualInfo_adaptiveIndex_cells(results_LNKS, keys, N_thrs, S_thrs):
    MIs = []
    MIs_bps = []
    AIs = []

    for key in keys:
        cell = results_LNKS.cells[key]
        stim = cell.stim
        fr = cell.fr
        contrast = cell.contrast
        theta = cell.LNKS_est['theta']
        theta_temp = _np.zeros(25)
        theta_temp[:8] = theta[:8]
        theta_temp[9:18] = theta[9:18]
        theta_temp[19:] = theta[19:]
        MIs_fr, MIs_fr_bps, AIs_fr = mutualInfo_adaptiveIndex(N_thrs, S_thrs, theta_temp, stim, contrast, fr)

        MIs.append(MIs_fr)
        MIs_bps.append(MIs_fr_bps)
        AIs.append(AIs_fr)

    return MIs, MIs_bps, AIs




