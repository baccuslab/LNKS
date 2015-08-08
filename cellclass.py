'''
cellclass.py

author: Bongsoo Suh
created: 2015-02-19

(C) 2015 bongsoos
'''
import numpy as _np
import scipy.io as _sio
from matplotlib import pyplot as _plt
from scipy.signal import find_peaks_cwt as _fpk
from peakdetect import peakdetect as _pd
import dataprocesstools as _dpt
import optimizationtools as _ot
import lnkstools as _lnks
import pickle


class Cell:
    '''
    Cell class for contrast adaptation experiment

    Usage
    -----
    In [0]: g1 = Cell()
    In [1]: g1.loadcell(options)  [options = {'filename', 'fc', 'threshold', 'filt_len', 'num_rep', 'gwin_len', 'gwin_std', 'apbflag'}]


    Methods
    -------
        loadcell:
            loads recording data(.mat files), given options

        LNKS:
            computes LNKS model of the cell

        LNK:
            computes LNK model of the cell

        Spiking:
            computes Spiking model of the cell

        fit:
            Fit model to the cell data given the objective and the model

        predict:
            Predict model estimate given the model and the parameters

        saveresult:
            save the optimization result to a file

        loadresult:
            laod the optimized result from a file

    Values
    ------
        st (ndarray):
            visual stimulus(averaged)

        mp (ndarray):
            membrane potential

        sp (ndarray):
            intracellular recording(membrane potential with spikes)

        spike (ndarray):
            spikes

        fr (ndarray):
            firing rates

        stims (ndarray):
            visual stimulus, 3 trials (3xN, N=300,000)

        mps (ndarray):
            membrane potentials, 3 trials (3xN, N=300,000)

        spikes (ndarray):
            spikes, 3 trials (3xN, N=300,000)

        frs (ndarray):
            firing rates, 3 trials (3xN, N=300,000)

    '''
    def __init__(self):
        '''
        Constructor:
            Creates cell object, and initializes hidden variables
                _num_rec (int):
                    Number of recording, or repeats of original experiment
                _num_apbrec (int):
                    Number of recording, or repeats of apb experiment
                _datalen (int):
                    Total length of the data, or the experiment
        '''

        self._num_rec = 0
        self._num_apbrec = 0
        self._datalen = 0



    def loadcell(self, options):
        '''
        load stimulus, cell response data from .mat file.

        Input
        -----
        options (dictionary):
            dictionary containing variables for loading data.
        '''
        self._importMatFile(options['filename'])
        self._set_mp(options['apbflag'])
        self._set_stim(options['apbflag'])
        self._set_resp(options['apbflag'])
        self._set_spikes(options['threshold'], options['apbflag'])
        self._filtering(options['fc'], options['filt_len'], options['num_rep'], options['apbflag'])
        self._set_fr(options['gwin_len'], options['gwin_std'])
        self._filtering_avg(options['fc'], options['filt_len'], options['num_rep'], options['apbflag'])
        self._set_contrast()

    def LNKS(self, param):
        '''
        Computes LNKS model of the cell
        '''

        st = self.st
        st = st - _np.min(st)
        st = st / _np.max(st)
        st = st - _np.mean(st)

        mp = self.mp
        mp = mp - _np.min(mp[20000:])
        mp = mp / _np.max(mp[20000:])

        fr = self.fr
        fr = fr / _np.max(fr)

        f, g, u, thetaK, X, v, r = _lnks.LNKS(param, st)

        v = v - _np.min(v[20000:])
        v = v / _np.max(v[20000:])

        self.LNKS_est = {
                'f': f,
                'g': g,
                'u': u,
                'thetaK': thetaK,
                'X': X,
                'v': mp,
                'v_est': v,
                'r': fr,
                'r_est': r,
                }


    def LNK(self, param):
        '''
        Computes LNK model of the cell
        '''

        st = self.st
        st = st - _np.min(st)
        st = st / _np.max(st)
        st = st - _np.mean(st)

        mp = self.mp
        mp = mp - _np.min(mp[20000:])
        mp = mp / _np.max(mp[20000:])

        f, g, u, thetaK, X, v = _lnks.LNK(param, st)

        self.LNK_est = {
                'f': f,
                'g': g,
                'u': u,
                'thetaK': thetaK,
                'X': X,
                'v_est': v,
                'v': mp,
                }


    def Spiking(self, f, f_get_h, f_gain, theta):
        '''
        Computes Spiking model of the cell given the model f and theta

        Input
        -----
        f (function):
            The spiking model

        f_get_h (function):
            Get the internal variable h of the spiking model

        f_gain (function):
            Get the instantaneous gain of the spiking model

        theta (ndarray):
            The model parameter

        Output
        ------
        y (ndarray):
            The output of the spiking model (firing rate)

        h (ndarray):
            The internal variable h, which is the membrane potential added with feedback

        gain (ndarray):
            The instantaneous gain
        '''
        mp = self.mp - _np.min(self.mp)
        mp = mp / _np.max(mp)

        self.est = f(theta, mp)

        ### need to be implemented


    def fit(self, fobj, f, theta, model, bnds=None, num_trials=1):
        '''
        Fit model to the cell data

        Input
        -----
        options (dictionary):
            'fobj':
            'f':
            'theta':
            'model':
        '''
        if model.lower() in ["spiking", "sci", "sc", "scif", "scf", "scie", "sc1d", "scief", "sdf", "scif2", "sc1df"]:

            mp = self.mp - _np.min(self.mp)
            mp = mp / _np.max(mp)
            fr = self.fr / _np.max(self.fr)

            data = (mp, fr)

            self.result = _ot.optimize(fobj, f, theta, data, bnds, True, num_trials)

        elif model.lower() == "lnks":
            st = self.st - _np.min(self.st)
            st = st / _np.max(st)
            st = st - _np.mean(st)
            fr = self.fr / _np.max(self.fr)

            data = (st, fr)

            self.result = _ot.optimize(fobj, f, theta, data, bnds, False, num_trials)

        elif model.lower() == "lnk":
            st = self.st - _np.min(self.st)
            st = st / _np.max(st)
            st = st - _np.mean(st)
            mp = self.mp - _np.min(self.mp)
            mp = mp / _np.max(mp)

            data = (st, mp)

            self.result = _ot.optimize(fobj, f, theta, data, bnds, False, num_trials)

        elif model.lower() == "lnk_est":
            mp = self.v_est
            fr = self.fr / _np.max(self.fr)

            data = (mp, fr)

            self.result = _ot.optimize(fobj, f, theta, data, bnds, True, num_trials)

        else:
            print("model name error")


    def predict(self, f, theta, model):
        '''
        Predict model estimate
        '''
        if model.lower() in ["spiking", "sci", "sc", "scif", "scf", "scie", "sc1d", "scief", "sdf", "scif2", "sc1df"]:

            mp = self.mp - _np.min(self.mp)
            mp = mp / _np.max(mp)
            fr = self.fr / _np.max(self.fr)

            data = (mp, fr)

            self.est = f(theta, data[0])

        elif model.lower() == "lnks":
            st = self.st - _np.min(self.st)
            st = st / _np.max(st)
            st = st - _np.mean(st)
            fr = self.fr / _np.max(self.fr)

            data = (st, fr)

            l, g, u, thetaK, X, v, r = _lnks.LNKS(theta, data[0])
            self.r_est = r
            self.v_est = v

        elif model.lower() == "lnk":
            st = self.st - _np.min(self.st)
            st = st / _np.max(st)
            st = st - _np.mean(st)
            mp = self.mp - _np.min(self.mp)
            mp = mp / _np.max(mp)

            data = (st, mp)

            l, g, u, thetaK, X, v = f(theta, data[0])
            self.v_est = v
            self.X_est = X
            self.u_est = u
            self.g_est = g
            self.thetaK = thetaK

        else:
            print("model name error")


    def saveresult(self, filename):
        '''
        save result to a file
        '''
        fileObject = open(filename, 'wb')
        pickle.dump(self.result, fileObject)
        fileObject.close()

    def loadresult(self, filename):
        '''
        load results to cell.result
        '''
        fileObject = open(filename, 'rb')
        self.result = pickle.load(fileObject)
        fileObject.close()


    def _importMatFile(self, filename):
        '''
        Imports data by loading Matlab .mat files

        Input
        -----
        filename (str):
            The .mat file name string should be given "string.mat"

        Output (hidden variables)
        -------------------------
        _num_rec (int):
        _num_apbrec (int):
        _data_len (int):
        _stim_list (list):
        _mp_list (list):
        _resp_list (list):
        _data (array list):

        '''
        mat = _sio.loadmat(filename)
        textdata = mat['textdata']
        textdata = textdata[0]
        data = mat['data']
        colheader = mat['colheaders']

        count = 0
        dic = {'mp':0, 'apb':0}
        stim_list = []
        resp_list = []
        mp_list = []
        for item in textdata:
            if 'apb' in item[0].lower():
                if 'mp' in item[0]:
                    dic['apb'] += 1
                    mp_list.append(('apb'+item[0][-1],count))
                if 'st' in item[0]:
                    stim_list.append(('apb'+item[0][-1], count))
                if 'resp' in item[0]:
                    resp_list.append(('apb'+item[0][-1], count))
            else:
                if ('mp' in item[0]):
                    dic['mp'] += 1
                    mp_list.append((item[0][-1],count))
                if 'st' in item[0]:
                    stim_list.append((item[0][-1], count))
                if 'resp' in item[0]:
                    resp_list.append((item[0][-1], count))

            count += 1

        self._textdata = textdata
        self._num_rec = dic['mp']
        self._num_apbrec = dic['apb']
        self._stim_list = stim_list
        self._mp_list = mp_list
        self._resp_list = resp_list
        self._datalen = data.shape[0]
        self._data = data

    def _set_stim(self, apbflag=False):
        '''
        Creates numpy array of stimuli

        Input
        -----
        apbflag (bool):
            Flag to indicate whether to load the APB or the original experiment
            'True' loads APB data. Default setting is False


        Output
        ------
        stims (ndarray):
            stimulus matrix (num_rec by datalen)
        '''
        if apbflag:
            self.stims = _np.zeros([self._num_apbrec, self._datalen])
            for i in range(self._num_apbrec):
                temp = [item for item in self._stim_list if item[0] == ("apb"+str(i+1))]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.stims[i,:] = _np.array(tempdata.T)

            self.st = _np.sum(self.stims, 0) / self._num_apbrec

        else:
            self.stims = _np.zeros([self._num_rec, self._datalen])
            for i in range(self._num_rec):
                temp = [item for item in self._stim_list if item[0] == str(i+1)]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.stims[i,:] = _np.array(tempdata.T)

            self.st = _np.sum(self.stims, 0) / self._num_rec


    def _set_contrast(self, section_length=20000):

        mu = _np.mean(self.st)

        num_section = _np.int(_np.floor(self.st.size / section_length))
        contrast = _np.zeros(num_section)

        for i in range(num_section):
            st = self.st[i*section_length : (i+1)*section_length]
            contrast[i] = _np.std(st) / mu

        self.contrast = contrast


    def _set_mp(self, apbflag=False):
        '''
        Creates numpy array of subthreshold membrane potential

        Input
        -----
        apbflag (bool):
            Flag to indicate whether to load the APB or the original experiment
            'True' loads APB data. Default setting is False

        Output
        ------
        mps (ndarray):
            subthreshold membrane potential matrix (num_rec by datalen)
        '''
        if apbflag:
            self.mps = _np.zeros([self._num_apbrec, self._datalen])
            for i in range(self._num_apbrec):
                temp = [item for item in self._mp_list if item[0] == ("apb"+str(i+1))]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.mps[i,:] = _np.array(tempdata.T)

            self.mp = _np.sum(self.mps, 0) / self._num_apbrec

        else:
            self.mps = _np.zeros([self._num_rec, self._datalen])
            for i in range(self._num_rec):
                temp = [item for item in self._mp_list if item[0] == str(i+1)]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.mps[i,:] = _np.array(tempdata.T)

            self.mp = _np.sum(self.mps, 0) / self._num_rec

    def _set_resp(self, apbflag=False):
        '''
        Creates numpy array of membrane potential recording

        Input
        -----
        apbflag (bool):
            Flag to indicate whether to load the APB or the original experiment
            'True' loads APB data. Default setting is False

        Output
        ------
        resps (ndarray):
            subthreshold membrane potential matrix (num_rec by datalen)
        '''
        if apbflag:
            self.resps = _np.zeros([self._num_apbrec, self._datalen])
            for i in range(self._num_apbrec):
                temp = [item for item in self._resp_list if item[0] == ("apb"+str(i+1))]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.resps[i,:] = _np.array(tempdata.T)

            self.sp = _np.sum(self.resps, 0) / self._num_apbrec

        else:
            self.resps = _np.zeros([self._num_rec, self._datalen])
            for i in range(self._num_rec):
                temp = [item for item in self._resp_list if item[0] == str(i+1)]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.resps[i,:] = _np.array(tempdata.T)

            self.sp = _np.sum(self.resps, 0) / self._num_rec


    def _set_spikes(self, threshold, apbflag):
        '''
        Set spikes given membrane potential response(original intracellular recording data),
        subthreshold membrane potential, and threshold.

        Input
        -----
        threshold:

        Output
        ------
        spikes:

        '''
        if apbflag:
            N = self._num_apbrec
        else:
            N = self._num_rec

        self.spikes = _np.zeros([N, self._datalen])
        for i in range(N):
            diff = self.resps[i,:] - self.mps[i,:]
            diff[diff<threshold] = 0
            #locs = _fpk(diff, _np.arange(1,2))
            _max, _min = _pd(diff, None, 2)
            locs = [p[0] for p in _max]
            self.spikes[i,locs] = 1

        self.spike = _np.sum(self.spikes, 0) / N


    def _set_fr(self, gwin_len, gwin_std):
        '''
        Sets firing rate given spikes and Gaussian window variables gwin_len, gwin_std.

        Input
        -----
        gwin_len (int):
            The length of the Gaussian window

        gwin_std (double):
            Standard deviation of the Gaussian window

        Output
        ------
        fr (ndarray):
            Firing rate
        '''
        self.fr = _dpt.Spk2FR(self.spikes, gwin_len, gwin_std)
        self.fr = 1000 * self.fr

        self.frs = _np.zeros(self.spikes.shape)
        for i in range(self.frs.shape[0]):
            self.frs[i,:] = _dpt.Spk2FR_1(self.spikes[i,:], gwin_len, gwin_std)
            self.frs[i,:] = 1000 * self.frs[i,:]

    def _filtering(self, fc, filt_len, num_rep, apbflag):
        '''
        Filtering process of the membrane potential recording to remove high frequency and dc noise.

        Input
        -----
        fc (double):
            Cut off frequency of the high pass filter.

        filt_len (int):
            Length of low pass smoothing filter

        num_rep (int):
            Number of repeats of low pass filtering
        '''
        if apbflag:
            N = self._num_apbrec
        else:
            N = self._num_rec

        for i in range(N):
            temph = _dpt.highpassfilter(self.mps[i,:], fc)
            templ = _dpt.lowpassfilter(_np.real(temph), filt_len, num_rep)
            self.mps[i,:] = _np.real(templ)

    def _filtering_avg(self, fc, filt_len, num_rep, apbflag):
        temph = _dpt.highpassfilter(self.mp, fc)
        templ = _dpt.lowpassfilter(_np.real(temph), filt_len, num_rep)
        self.mp = _np.real(templ)

    def reshape3D(self):
        self.st     = self.st.reshape(1, -1, 1)
        self.spike  = self.spike.reshape(1, -1, 1)
        self.mp     = self.mp.reshape(1, -1, 1)


def main():
    filename = 'g12.mat'
    options = {'filename': filename, 'fc': 3, 'threshold': 1, 'filt_len': 10, 'num_rep': 5, 'gwin_len': 1000, 'gwin_std': 10, 'apbflag': False}

    g = Cell()
    g.loadcell(options)

    print(g.fr.size)
    _plt.plot(g.fr)
    _plt.show()


