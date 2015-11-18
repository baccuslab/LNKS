'''
analysistools.py

Tools for analyzing cell data

* Linear Nonlinear(LN) model

author: Bongsoo Suh
created: 2015-03-08

(C) 2015 bongsoos
'''

import numpy as _np
from scipy import interpolate as _interp

'''
    Linear-Nonlinear(LN) Model
'''
def LN(x, y, num_bins=30, M=1000):
    '''
    Return LN model(Linear filter and Nonlinearity(sigmoidal function))

    Input
    -----
    x (ndarray):
        input data to the LN model

    y (ndarray):
        output data of the LN model

    M (int):
        linear filter(kernel) length

    Output
    ------
    LN_est (ndarray):

    L_est (ndarray):

    f (ndarray):
        Linear filter of the LN model

    theta (ndarray):
        Parameters of sigmoidal function

    result (dictionary):
        Optimization result
    '''

    N = min(x.size, y.size)

    x = x[:N]
    y = y[:N]

    x_mean = _np.mean(x)
    x = x - x_mean
    x_std = _np.std(x)

    y_mean = _np.mean(y)
    _y = y - y_mean
    f = RC(x, _y, M)
    f = f - _np.mean(f[400:])

    L_est = _np.convolve(x, f)
    L_est = L_est[:N]
    f = f / (_np.std(L_est)/x_std*x_mean)
    f = f - _np.mean(f[500:])
    L_est = _np.convolve(x, f)
    L_est = L_est[:N]

    LN_est, snl, snlx = SNL(L_est, _y, num_bins)
    snl = snl + y_mean
    LN_est = LN_est + y_mean

    return LN_est, L_est, f, snl, snlx




def RC(x, y, M):
    '''
    Return Reverse correlation of x and y

    Input
    -----
    x (ndarray):
        input data to the LN model

    y (ndarray):
        output data of the LN model

    m (int):
        kernel length

    Output
    ------
    f (ndarray):
        Reverse correlation (Linear filter of LN model)
    '''

    N = min(x.size, y.size)
    x = x[:N]
    y = y[:N]
    x = x - _np.mean(x)
    y = y - _np.mean(y)

    offset = 100
    num_pers = _np.int(_np.floor((N-M)/offset))

    f = _np.zeros(M)
    fft_f = _np.zeros(M)
    cross_xy = fft_f
    denom = cross_xy

    for i in range(num_pers):
        x_per = x[i*offset:i*offset + M]
        y_per = y[i*offset:i*offset + M]

        auto_x = _np.abs(_np.fft.fft(x_per))**2
        auto_y = _np.abs(_np.fft.fft(y_per))**2

        cross_xy = cross_xy + _np.conjugate(_np.fft.fft(x_per)) * _np.fft.fft(y_per)
        denom = denom + auto_x + _np.mean(auto_y)*10

    fft_f = cross_xy / denom
    f = _np.real(_np.fft.ifft(fft_f))

    return f

def SNL(linear_est, resp, num_bins):
    '''
    Static Nonlinearity
    '''
    N = min(linear_est.size, resp.size)
    linear_est = linear_est[:N]
    resp = resp[:N]

    snl = _np.zeros(num_bins)
    snlx = _np.zeros(num_bins)

    bin_length = _np.floor(N / num_bins)
    last_bin = N - bin_length * (num_bins - 1)

    filt = _np.ones(bin_length) / bin_length

    lest_sorted = _np.sort(linear_est)
    sorted_index = _np.argsort(linear_est)
    resp_sorted = resp[sorted_index]

    pre_snlx = _np.convolve(lest_sorted, filt)
    pre_snl = _np.convolve(resp_sorted, filt)

    pre_snlx = pre_snlx[:N]
    pre_snl = pre_snl[:N]

    snlx[:-1] = pre_snlx[bin_length-1:N - last_bin:bin_length]
    snl[:-1] = pre_snl[bin_length-1:N - last_bin:bin_length]
    snlx[-1] = _np.mean(pre_snlx[N - last_bin:])
    snl[-1] = _np.mean(pre_snl[N - last_bin:])

    fsnl = _interp.interp1d(snlx, snl, bounds_error=False)
    # fsnl = _interp.interp1d(snlx, snl)

    snlx = _np.linspace(_np.min(snlx), _np.max(snlx), num_bins)
    snl = fsnl(snlx)

    LN_est = SNL_eval(fsnl, linear_est, snlx, snl)

    return LN_est, snl, snlx


def SNL_eval(fsnl, linear_est, snlx, snl):
    delta_bin = snlx[1] - snlx[0]
    num_bins = snlx.size
    index1 = _np.zeros(linear_est.shape)
    index2 = _np.zeros(linear_est.shape)

    index1 = _np.floor((linear_est - _np.min(snlx)) / delta_bin) + 1
    index1 = ((index1 >= 1)*(index1 <=num_bins)) * index1 + (index1 < 1) + (index1 > num_bins)*num_bins

    index2 = index1 + ((index1 > 1) * (index1 < num_bins))

    LN_est = snl[index1] + ((linear_est - snlx[index1])/delta_bin) * (snl[index2]-snl[index1])

    # LN_est = _np.zeros(linear_est.size)

    lest_sorted = _np.sort(linear_est)
    sorted_index = _np.argsort(linear_est)

    LN_est = fsnl(lest_sorted)
    LN_est = LN_est[sorted_index]
    for i in range(linear_est.size):
        LN_est[i] = fsnl(linear_est[i])

    return LN_est


