'''
dataprocesstools.py

Tools for processing data

author: Bongsoo Suh
created: 2015-02-19

(C) 2015 bongsoos
'''
import numpy as _np

def Spk2FR(spikes, gwin_len, gwin_std):
    '''
    Converts spikes to Firing Rate(FR) using Gaussian window smoothing.
    spikes contains multiple trials.

    Input
    -----
    spikes (ndarray):
        Spikes recording data. Array can be 2D, row is different repeat and col is recorded data.

    gwin_len (int):
        The length of the Gaussian window

    gwin_std (double):
        Standard deviation of the Gaussian window

    Output
    ------
    fr (ndarray):
        Firing rate
    '''
    numTrials = spikes.shape[0]
    gwin = gausswin(gwin_len, gwin_std)
    psth = _np.sum(spikes,0) / numTrials
    fr = _np.convolve(gwin, psth)
    fr = fr[gwin_len/2:]
    fr = fr[:spikes.shape[1]]

    return fr


def Spk2FR_1(spikes, gwin_len, gwin_std):
    '''
    Converts spikes to Firing Rate(FR) using Gaussian window smoothing.
    single trial

    Input
    -----
    spikes (ndarray):
        Spikes recording data. Array can be 2D, row is different repeat and col is recorded data.

    gwin_len (int):
        The length of the Gaussian window

    gwin_std (double):
        Standard deviation of the Gaussian window

    Output
    ------
    fr (ndarray):
        Firing rate
    '''
    gwin = gausswin(gwin_len, gwin_std)
    fr = _np.convolve(gwin, spikes)
    fr = fr[gwin_len/2:]
    fr = fr[:spikes.size]

    return fr


def gausswin(length, std):
    '''
    Returns a Gaussian window given length and standard deviation.

    Input
    -----
    length (int):
        The length of the Gaussian window

    std (double):
        Standard deviation of the Gaussian window

    Output
    ------
    gwin (ndarray):
        Gaussian window of length length and standard deviation std.
    '''
    x = _np.linspace(-length/2, length/2, length)

    gwin = (1/ (_np.sqrt(2*_np.pi)*std)) * _np.exp(-(x**2) / (2*(std**2)))

    return gwin

def highpassfilter(mp, fc):
    '''
    High pass filtering of the signal mp given the cut off frequency fc.

    Input
    -----
    mp (ndarray):
        membrane potential signal.

    fc (double):
        Cut off frequency of the high pass filter.

    Output
    ------
    mp_high (ndarray):
        high pass filtered mp signal

    '''
    L = mp.size
    N = 2^_nextpow2(L)
    X = _np.fft.fft(mp - _np.mean(mp), N)

    mask = _np.ones(X.size)
    mask[:fc+1] = 0
    mask[-fc:] = 0

    mp_high = _np.fft.ifft(X*mask, N) 
    mp_high = mp_high[:mp.size]

    return mp_high


def _nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n

def lowpassfilter(mp, filt_len, num_rep):
    '''
    Filtering process of the membrane potential recording to remove high frequency and dc noise.

    Input
    -----
    mp (ndarray):
        membrane potential signal.

    filt_len (int):
        Length of low pass smoothing filter

    num_rep (int):
        Number of repeats of low pass filtering

    Output
    ------
    mp_low (ndarray):
        low pass filtered mp signal
    '''
    mp_low = mp - _np.mean(mp)
    filt = _np.ones(filt_len)/filt_len

    for i in range(num_rep):
        mp_low = _np.convolve(filt, mp_low)

    mp_low = mp_low[num_rep*(filt_len/2):]
    mp_low = mp_low[:mp.size]

    return mp_low


