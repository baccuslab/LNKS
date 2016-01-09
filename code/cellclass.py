'''
cellclass.py

author: Bongsoo Suh
created: 2015-02-19

(C) 2015 bongsoos
'''
import numpy as _np
import scipy.io as _sio
from scipy.signal import find_peaks_cwt as _fpk
from peakdetect import peakdetect as _pd
import dataprocesstools as _dpt
import optimizationtools as _ot
import lnkstools as _lnks
import spikingblocks as _sb
import spikingtools as _st
import pickle
import pdb

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
        stim (ndarray):
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

    def LNKS(self):
        '''
        Computes LNKS model of the cell
        '''

        if self.result and 'theta' in self.result:
            param = self.result['theta']
        else:
            raise NameError('result is not defined. load result using loadresult method.')

        pathway = get_pathway(param, 'LNKS')
        w = get_weights(param, 'LNKS')

        st = self.stim
        st = normalize_stim(st)

        mp = self.mp
        mp = normalize_stim(mp)

        fr = self.fr
        fr = fr / _np.max(fr)

        f, g, u, thetaK, X, v, r = _lnks.LNKS(param, st, pathway)

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
            'w': w,
            }

        if pathway == 2:
            self.LNKS_est['v_est_on'] = X[0][1,:] * w[0]
            self.LNKS_est['v_est_off'] = X[1][1,:] * w[1]


    def LNK(self):
        '''
        Computes LNK model of the cell
        '''

        if self.result and 'theta' in self.result:
            param = self.result['theta']
        else:
            raise NameError('result is not defined. load result using loadresult method.')

        pathway = get_pathway(param, 'LNK')
        w = get_weights(param, 'LNK')

        st = self.stim
        st = normalize_stim(st)

        mp = self.mp
        # mp = normalize_0_1(mp)
        mp = normalize_stim(mp)

        f, g, u, thetaK, X, v = _lnks.LNK(param, st, pathway)

        self.LNK_est = {
            'f': f,
            'g': g,
            'u': u,
            'thetaK': thetaK,
            'X': X,
            'v_est': v,
            'v': mp,
            'w': w,
            }

        if pathway == 2:
            self.LNK_est['v_est_on'] = X[0][1,:] * w[0]
            self.LNK_est['v_est_off'] = X[1][1,:] * w[1]


    def Spiking(self, theta, model):
        '''
        Computes Spiking model of the cell given the model f and theta

        Output
        ------
        y (ndarray):
            The output of the spiking model (firing rate)

        m (ndarray):

        h (ndarray):
            The internal variable h, which is the membrane potential added with feedback

        gain (ndarray):
            The instantaneous gain

        fb (ndarray):

        b (ndarray):
        '''
        if model.lower() in ['lnks', 'lnks_mp']:
            mp = self.LNKS_est['v_est']

        elif model.lower() == 'spiking':
            mp = normalize_stim(self.mp)

        else:
            raise ValueError('model error')

        dv = _sb.deriv(mp, 0.001)
        m = _st.SC1DF_get_m(theta, mp, dv)
        r, h, gain, fb, b = _sb.SC1DF(theta, mp)

        self.Spiking_est = {
            'r': normalize_0_1(self.fr),
            'r_est': r,
            'm': m,
            'h': h,
            'gain': gain,
            'fb': fb,
            'b': b,
        }


    def summary(self, model):
        '''
        display model fit summary

        Input
        -----
        model (string): LNK, LNKS, Spiking
        '''
        def _get_nonlinearity_param(thetaN):
            print("%45s" %("Nonlinearity Parameters"))
            print("-"*70)
            print("%16s" %(" a * (g(t) + thr)"))
            print("%6.3f*(g(t)+%6.3f)" %(thetaN[1], thetaN[0]/thetaN[1]))
            print("\n")

        def _get_kinetics_ratio(thetaK):
            print("%45s" %("Kinetics Parameters"))
            print("-"*70)
            print("%15s %10s %10s %10s %20s" %("ka1*u(t) + ka2", "kfi", "kfr", "ksi", "ksr1*u(t)+ksr2"))
            print("%5.2fu(t)+%5.2f %10.2f %10.2f %10.2f %10.2fu(t)+%5.2f" %(thetaK[0], thetaK[6], thetaK[1], thetaK[2], thetaK[3], thetaK[4], thetaK[5]))
            print("\n")
            print("%10s %10s" %("kfi/ka", "kfr/ka"))
            print("%10.3f %10.3f" %(thetaK[1]/thetaK[0], thetaK[2]/thetaK[0]))
            print("="*70)
            print("\n")

        if model == 'LNKS' and self.result and self.LNKS_est:
            param = self.result['theta']
            pathway = get_pathway(param, model)

            if pathway == 1:
                _get_kinetics_ratio(self.LNKS_est['thetaK'])

            elif pathway == 2:
                print("%45s" %("On Pathway"))
                _get_kinetics_ratio(self.LNKS_est['thetaK'][0])
                print("%45s" %("Off Pathway"))
                _get_kinetics_ratio(self.LNKS_est['thetaK'][1])
                print("%45s" %("Weights"))
                w = get_weights(param, model)
                print("%10s %10s" %("w_on", "w_off"))
                print("%10.3f %10.3f" %(w[0], w[1]))

        elif model == 'LNK' and self.result and self.LNK_est:
            param = self.result['theta']
            pathway = get_pathway(param, model)

            if pathway == 1:
                thetaN = self.result['theta'][8:10]
                _get_nonlinearity_param(thetaN)
                _get_kinetics_ratio(self.LNK_est['thetaK'])

            elif pathway == 2:
                print("%45s" %("On Pathway"))
                _get_kinetics_ratio(self.LNK_est['thetaK'][0])
                print("%45s" %("Off Pathway"))
                _get_kinetics_ratio(self.LNK_est['thetaK'][1])
                print("%45s" %("Weights"))
                w = get_weights(param, model)
                print("%10s %10s" %("w_on", "w_off"))
                print("%10.3f %10.3f" %(w[0], w[1]))

        else:
            raise NameError('%s_est is not defined. load result using loadresult method.' %(model))

        s = None
        return s



    def fit(self, fobj, f, theta, model, bnds=None, options=None):
        '''
        Fit model to the cell data

        Input
        -----
        options (dictionary):
            'fobj': objective function
            'f': model function
            'theta': model parameter
            'model': model type
            'bnds': model parameter boundary
            'options': optimization options
                'pathway': LNK pathway (1, 2)
                'crossval': cross-validation (True, False)
        '''

        data = get_data(self, model, options)
        self.result = _ot.optimize(fobj, f, theta, data, bnds, options)


    def predict(self, f, theta, model, pathway=None):
        '''
        Predict model estimate
        '''
        if model.lower() in ["spiking", "sci", "sc", "scif", "scf", "scie", "sc1d", "scief", "sdf", "scif2", "sc1df", "sc1df_1"]:
            '''
            Predicting Spiking Model from the membrane potential data
            '''

            mp = self.mp - _np.min(self.mp)
            mp = mp / _np.max(mp)
            fr = self.fr / _np.max(self.fr)

            data = (mp, fr)

            self.est = f(theta, data[0])

        elif model.lower() == "spiking_est":
            '''
            Predicting Spiking model from the LNK model estimate.
            '''
            mp = self.v_est
            fr = self.fr / _np.max(self.fr)

            data = (mp, fr)

            self.est = f(theta, data[0])

        elif model.lower() in ["lnks", "lnks_mp"]:
            '''
            Predicting LNKS model
            '''
            st = normalize_stim(self.stim)
            fr = self.fr / _np.max(self.fr)

            data = (st, fr)

            l, g, u, thetaK, X, v, r = _lnks.LNKS(theta, data[0], pathway)
            self.r_est = r
            self.v_est = v

        elif model.lower() == "lnk":
            '''
            Predicting LNK model
            '''
            st = normalize_stim(self.stim)
            mp = self.mp - _np.min(self.mp)
            mp = mp / _np.max(mp)

            data = (st, mp)

            l, g, u, thetaK, X, v = f(theta, data[0], pathway)
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
        # colheader = mat['colheaders']

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

            self.stim = _np.sum(self.stims, 0) / self._num_apbrec

        else:
            self.stims = _np.zeros([self._num_rec, self._datalen])
            for i in range(self._num_rec):
                temp = [item for item in self._stim_list if item[0] == str(i+1)]
                tempdata = _np.array(self._data[:,temp[0][1]])
                self.stims[i,:] = _np.array(tempdata.T)

            self.stim = _np.sum(self.stims, 0) / self._num_rec


    def _set_contrast(self, section_length=20000):

        mu = _np.mean(self.stim)

        num_section = _np.int(_np.floor(self.stim.size / section_length))
        contrast = _np.zeros(num_section)

        for i in range(num_section):
            st = self.stim[i*section_length:(i+1)*section_length]
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
            # locs = _fpk(diff, _np.arange(1,2))
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


def get_pathway(theta, model):
    '''
    return pathway
    '''
    if (model == 'LNKS' and theta.size == 43) or (model == 'LNK' and theta.size == 36):
        pathway = 2
    elif (model == 'LNKS' and theta.size == 25) or (model == 'LNK' and theta.size == 18):
        pathway = 1
    else:
        raise ValueError('param size error')

    return pathway

def get_weights(theta, model):
    '''
    return weights
    '''
    if (model == 'LNKS' and theta.size == 43) or (model == 'LNK' and theta.size == 36):
        w = [theta[17], theta[35]]
    elif (model == 'LNKS' and theta.size == 25) or (model == 'LNK' and theta.size == 18):
        w = theta[17]
    else:
        raise ValueError('param size error')

    return w



def get_data(cell, model, options):
    '''
    get data tuple for model optimization

    Input
    -----
    cell (cellclass)
    model (string)

    Output
    ------
    data (tuple)
    '''

    st = normalize_stim(cell.stim)
    mp = normalize_stim(cell.mp)
    # mp = normalize_0_1(cell.mp)
    fr = cell.fr / _np.max(cell.fr)

    if options['crossval']:
        st_train = st[:-20000]
        mp_train = mp[:-20000]
        fr_train = fr[:-20000]
    else:
        st_train = st
        mp_train = mp
        fr_train = fr

    if model.lower() in ["spiking", "spiking_thr", "sci", "sc", "scif", "scf", "scie", "sc1d", "scief", "sdf", "scif2", "sc1df", "sc1df_1"]:
        '''
        Fitting Spiking Blocks
        '''
        data = (mp_train, fr_train, options)

    elif model.lower() == "lnks":
        '''
        Fitting LNKS model using only firing rate as an output
        '''
        data = (st_train, fr_train, options)

    elif model.lower() == "lnks_mp":
        '''
        Fitting LNKS model using both membrane potential and firing rate as outputs
        '''
        outputs = [mp_train, fr_train]
        data = (st_train, outputs, options)

    elif model.lower() == "lnk":
        '''
        Fitting LNK model to the membrane potential
        '''
        data = (st_train, mp_train, options)

    elif model.lower() == "spiking_est":
        '''
        Fitting Spiking model to the firing rate using the LNK model estimate.
        '''
        mp = cell.v_est
        if options['crossval']:
            mp_train = mp[:-20000]
        else:
            mp_train = mp

        data = (mp_train, fr_train, options)

    else:
        raise ValueError('The model name is not appropriate.')

    return data


def normalize_stim(stim):
    '''
    Normalize stimulus to have zero mean and maximum difference between max and min to be 1.

    Input
    -----
    stim (ndarray): visual stimulus

    Output
    ------
    st (ndarray): normalized stimulus
    '''

    st = normalize_0_1(stim)
    st = st - _np.mean(st)

    return st

def normalize_0_1(x):
    '''
    Return x_norm, a normalized x to be between 0~1.
    '''
    x_norm = x - _np.min(x)
    x_norm = x_norm / _np.max(x_norm)

    return x_norm

def main():
    filename = 'g12.mat'
    options = {'filename': filename, 'fc': 3, 'threshold': 1, 'filt_len': 10, 'num_rep': 5, 'gwin_len': 1000, 'gwin_std': 10, 'apbflag': False}

    g = Cell()
    g.loadcell(options)

    print(g.fr.size)


