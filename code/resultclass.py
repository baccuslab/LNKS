'''
resultclass.py

author: Bongsoo Suh
created: 2015-12-02

usage:
    import resultclass as rc
    import loaddatatools as ldt
    cells = ldt.loadcells()
    keys = ['g9','g11','g12','g13']
    path = 'path/to/result/'

    results = rc.Results(cells, keys)
    results.load_results(path, 'LNK')
    results.compute_instant_gain('LNK')
    results.compute_average_gain('LNK')
    results.show_corrcoef()
    results.summary()
    results.get_summary()

(C) 2015 bongsoos
'''
import numpy as _np
import scipy.io as _sio
import dataprocesstools as _dpt
import optimizationtools as _ot
import lnkstools as _lnks
import spikingblocks as _sb
import cellclass as _ccls
import analysistools_bak as _at
import plottools as _pt
import plotcelltools as _pct
import pickle as _pickle
import pdb as _pdb
import pyio as _pyio
import pandas as _pd
import multiprocessing as _mult

class Results:
    '''
    Results class
    '''
    def __init__(self, cells, keys):
        '''
        Constructor:
            Cell object, and initializes hidden variables
        '''

        self.cells = cells
        self.keys = keys

    def load_results(self, dir_path, model):
        self._load_results(dir_path, model)
        self._compute_instant_gain(model)
        self._compute_average_gain(model)
        self._summary(model)

    def show_corrcoef(self):
        cc_list = [(key, self.cells[key].result['corrcoef']) for key in self.keys]
        cc_list.sort(key=lambda x:x[1])
        print("%6s %10.4s" %("cell","corrcoef"))
        for i in range(len(cc_list)):
            print("%6s %10.4f" %(cc_list[i][0], cc_list[i][1]))
        return

    def _load_results(self, dir_path, model):
        '''
        load results
        '''
        for key in self.keys:
            filename = dir_path + key + '_results.pickle'
            self.cells[key].loadresult(filename)

            contrast = self.cells[key].contrast
            idx_contrast = _np.argsort(contrast)
            x_contrast = contrast[idx_contrast]
            self.cells[key].x_contrast = x_contrast
            self.cells[key].idx_contrast = idx_contrast

            if model.lower() == "lnk":
                self.cells[key].LNK()
                self.cells[key].LNK_est['theta'] = self.cells[key].result['theta']
                self.cells[key].LNK_est['corrcoef'] = self.cells[key].result['corrcoef']
                self.cells[key].pathway = _ccls.get_pathway(self.cells[key].result['theta'], model)

            elif model.lower() in ["lnks", "lnks_mp"]:
                self.cells[key].LNKS()
                self.cells[key].LNKS_est['theta'] = self.cells[key].result['theta']
                self.cells[key].LNKS_est['corrcoef'] = self.cells[key].result['corrcoef']
                self.cells[key].pathway = _ccls.get_pathway(self.cells[key].result['theta'], model)

            elif model.lower() == "spiking":
                self.cells[key].Spiking(self.cells[key].result['theta'], 'spiking')
                self.cells[key].Spiking_est['theta'] = self.cells[key].result['theta']
                self.cells[key].Spiking_est['corrcoef'] = self.cells[key].result['corrcoef']

    def _compute_instant_gain(self, model):
        '''
        Compute instantaneous gains of the model
        '''
        for key in self.keys:
            if model.lower() == "lnk":
                theta = self.cells[key].LNK_est['theta']
                pathway = self.cells[key].pathway

                # Compute Instantaneous Gains
                # Gain: Nonlinearity
                u = self.cells[key].LNK_est['u']
                N_gain = get_N_gain(theta,u,pathway)
                self.cells[key].LNK_est['N_gain'] = N_gain

                # Gain: Kinetics
                X = self.cells[key].LNK_est['X']
                K_gain = get_K_gain(X,pathway)
                self.cells[key].LNK_est['K_gain'] = K_gain

                # Gain: Synaptic gain(NK Gain)
                NK_gain = get_NK_gain(N_gain, K_gain, pathway)
                self.cells[key].LNK_est['NK_gain'] = NK_gain


            elif model.lower() in ["lnks", "lnks_mp"]:
                theta = self.cells[key].LNKS_est['theta']

                pathway = _ccls.get_pathway(theta, model)

                # Compute Instantaneous Gains
                # Gain: Nonlinearity
                u = self.cells[key].LNKS_est['u']
                N_gain = get_N_gain(theta,u,pathway)
                self.cells[key].LNKS_est['N_gain'] = N_gain

                # Gain: Kinetics
                X = self.cells[key].LNKS_est['X']
                K_gain = get_K_gain(X,pathway)
                self.cells[key].LNKS_est['K_gain'] = K_gain

                # Gain: Synaptic gain(NK Gain)
                NK_gain = get_NK_gain(N_gain, K_gain, pathway)
                self.cells[key].LNKS_est['NK_gain'] = NK_gain

                if pathway == 1:
                    self.cells[key].Spiking(theta[18:], model)
                elif pathway == 2:
                    self.cells[key].Spiking(theta[36:], model)
                S_gain = self.cells[key].Spiking_est['gain']
                self.cells[key].LNKS_est['S_gain'] = S_gain

                KS_gain = get_KS_gain(K_gain, S_gain, pathway)
                self.cells[key].LNKS_est['KS_gain'] = KS_gain

                NKS_gain = get_NKS_gain(N_gain, K_gain, S_gain, pathway)
                self.cells[key].LNKS_est['NKS_gain'] = NKS_gain

            elif model.lower() == 'spiking':

                S_gain = self.cells[key].Spiking_est['gain']
                self.cells[key].Spiking_est['S_gain'] = S_gain

    def _compute_average_gain(self, model):
        '''
        Compute average gains of the model
        '''
        # IDX_OFFSET = 5000
        SECTION_LEN = 20000

        for key in self.keys:
            if model.lower() == "lnk":
                pathway = self.cells[key].pathway

                N_gain = self.cells[key].LNK_est['N_gain']
                K_gain = self.cells[key].LNK_est['K_gain']
                NK_gain = self.cells[key].LNK_est['NK_gain']

                idx_contrast = self.cells[key].idx_contrast

                if pathway == 1:
                    self.cells[key].LNK_est['G_N'] = _np.array([_np.mean(N_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNK_est['G_K'] = _np.array([_np.mean(K_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNK_est['G_NK'] = _np.array([_np.mean(NK_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])

                elif pathway == 2:
                    N_gain_on, N_gain_off = N_gain
                    K_gain_on, K_gain_off = K_gain
                    NK_gain_on, NK_gain_off = NK_gain
                    G_N_on = _np.array([_np.mean(N_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_K_on = _np.array([_np.mean(K_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_NK_on = _np.array([_np.mean(NK_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_N_off = _np.array([_np.mean(N_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_K_off = _np.array([_np.mean(K_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_NK_off = _np.array([_np.mean(NK_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNK_est['G_N'] = [G_N_on, G_N_off]
                    self.cells[key].LNK_est['G_K'] = [G_K_on, G_K_off]
                    self.cells[key].LNK_est['G_NK'] = [G_NK_on, G_NK_off]

            elif model.lower() in ["lnks", "lnks_mp"]:
                pathway = self.cells[key].pathway

                N_gain  = self.cells[key].LNKS_est['N_gain']
                K_gain  = self.cells[key].LNKS_est['K_gain']
                NK_gain = self.cells[key].LNKS_est['NK_gain']
                S_gain  = self.cells[key].LNKS_est['S_gain']
                KS_gain = self.cells[key].LNKS_est['KS_gain']
                NKS_gain = self.cells[key].LNKS_est['NKS_gain']

                idx_contrast = self.cells[key].idx_contrast

                if pathway == 1:
                    self.cells[key].LNKS_est['G_N'] = _np.array([_np.mean(N_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNKS_est['G_K'] = _np.array([_np.mean(K_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNKS_est['G_NK'] = _np.array([_np.mean(NK_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNKS_est['G_S'] = _np.array([_np.mean(S_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNKS_est['G_KS'] = _np.array([_np.mean(KS_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNKS_est['G_NKS'] = _np.array([_np.mean(NKS_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])

                elif pathway == 2:
                    N_gain_on, N_gain_off = N_gain
                    K_gain_on, K_gain_off = N_gain
                    NK_gain_on, NK_gain_off = N_gain
                    S_gain_on, S_gain_off = S_gain
                    KS_gain_on, KS_gain_off = KS_gain
                    NKS_gain_on, NKS_gain_off = NKS_gain
                    G_N_on = _np.array([_np.mean(N_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_K_on = _np.array([_np.mean(K_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_NK_on = _np.array([_np.mean(NK_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_N_off = _np.array([_np.mean(N_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_K_off = _np.array([_np.mean(K_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_NK_off = _np.array([_np.mean(NK_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_S_on = _np.array([_np.mean(S_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_KS_on = _np.array([_np.mean(KS_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_NKS_on = _np.array([_np.mean(NKS_gain_on[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_S_off = _np.array([_np.mean(S_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_KS_off = _np.array([_np.mean(KS_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    G_NKS_off = _np.array([_np.mean(NKS_gain_off[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]) for i in range(len(idx_contrast))])
                    self.cells[key].LNK_est['G_N'] = [G_N_on, G_N_off]
                    self.cells[key].LNK_est['G_K'] = [G_K_on, G_K_off]
                    self.cells[key].LNK_est['G_NK'] = [G_NK_on, G_NK_off]
                    self.cells[key].LNK_est['G_S'] = [G_S_on, G_S_off]
                    self.cells[key].LNK_est['G_KS'] = [G_KS_on, G_KS_off]
                    self.cells[key].LNK_est['G_NKS'] = [G_NKS_on, G_NKS_off]

            elif model.lower() == 'spiking':
                idx_contrast = self.cells[key].idx_contrast
                S_gain = self.cells[key].Spiking_est['S_gain']
                G_S = _np.zeros(len(idx_contrast))
                for i in range(len(idx_contrast)):
                    S_gain_ = S_gain[idx_contrast[i]*SECTION_LEN:(idx_contrast[i]+1)*SECTION_LEN]
                    G_S[i] = _np.mean(S_gain_)
                self.cells[key].Spiking_est['G_S'] = G_S

    def _summary(self, model='LNK'):
        '''
        get model parameter summary

        Input
        -----
        model (string): LNK, LNKS, Spiking
        '''

        if model.lower() in ['lnk','lnks']:
            cols = ["cell","type","corr", "N_slope_off", "N_thr_off", "Kfi_Ka_off", "Kfr_Ka_off", "W_off", "N_slope_on", "N_thr_on", "Kfi_Ka_on", "Kfr_Ka_on", "W_on"]
            summary = _pd.DataFrame(columns=cols)

            for key in self.keys:
                if model.lower() == 'lnks':
                    theta = self.cells[key].LNKS_est['theta']
                    corrcoef = self.cells[key].LNKS_est['corrcoef']
                elif model.lower() == 'lnk':
                    theta = self.cells[key].LNK_est['theta']
                    corrcoef = self.cells[key].LNK_est['corrcoef']
                pathway = self.cells[key].pathway
                K_ratios = get_kinetics_ratio(theta, pathway)
                W = _ccls.get_weights(theta, model)
                N_thetas = get_nonlinearity_param(theta, pathway)

                if pathway == 1:
                    summary.loc[len(summary)] = [key,1,corrcoef,N_thetas[0],N_thetas[1],K_ratios[0],K_ratios[1],W,_np.nan,_np.nan,_np.nan,_np.nan,_np.nan]
                elif pathway == 2:
                    summary.loc[len(summary)] = [key,2,corrcoef,N_thetas[1][0],N_thetas[1][1],K_ratios[1][0],K_ratios[1][1],W[1], N_thetas[0][0],N_thetas[0][1],K_ratios[0][0],K_ratios[0][1],W[0]]

            self.summary = summary

        elif model.lower() == 'spiking':
            cols = ["cell","type","corr", "theta_0", "theta_v", "theta_dv", "fb_amp_1", "fb_tau_1", "fb_amp_2", "fb_tau2"]
            summary = _pd.DataFrame(columns=cols)

            for key in self.keys:
                theta = self.cells[key].Spiking_est['theta']
                corrcoef = self.cells[key].Spiking_est['corrcoef']
                pathway = None

                summary.loc[len(summary)] = [key,pathway,corrcoef,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6]]

            self.summary = summary

    def get_summary(self):
        '''
        display model fit summary

        Input
        -----
        model (string): LNK, LNKS, Spiking
        '''
        return self.summary

    def show_summary(self):
        '''
        display model fit summary

        Input
        -----
        model (string): LNK, LNKS, Spiking
        '''
        print(self.summary)


def get_N_gain(theta, u, pathway):
    '''
    Return instantaneous gain of the Nonlinearity
    '''
    if pathway == 1:
        return u * (1-u) * theta[9]

    elif pathway == 2:
        return [u[0] * (1-u[0]) * theta[9], u[1] * (1-u[1]) * theta[27]]


def get_K_gain(X, pathway):
    '''
    Return instantaneous gain of the Kinetics block
    '''
    if pathway == 1:

        return X[0]

    elif pathway == 2:
        return [X[0][0], X[1][0]]

def get_NK_gain(N_gain, K_gain, pathway):
    '''
    Return synaptic gain (N_gain * K_gain)
    '''
    if pathway == 1:
        return _ccls.normalize_0_1(N_gain * K_gain)

    elif pathway == 2:
        return [_ccls.normalize_0_1(N_gain[i] * K_gain[i]) for i in range(2)]

def get_KS_gain(K_gain, S_gain, pathway):
    '''
    Return synaptic gain (N_gain * K_gain)
    '''
    if pathway == 1:
        return _ccls.normalize_0_1(K_gain * S_gain)

    elif pathway == 2:
        return [_ccls.normalize_0_1(K_gain[i] * S_gain[i]) for i in range(2)]

def get_NKS_gain(N_gain, K_gain, S_gain, pathway):
    '''
    Return synaptic gain (N_gain * K_gain)
    '''
    if pathway == 1:
        # return _ccls.normalize_0_1(N_gain * K_gain * S_gain)
        return N_gain * K_gain * S_gain

    elif pathway == 2:
        # return [_ccls.normalize_0_1(N_gain[i] * K_gain[i] * S_gain[i]) for i in range(2)]
        return [N_gain[i] * K_gain[i] * S_gain[i] for i in range(2)]

def get_kinetics_ratio(theta, pathway):
    '''
    get kinetics inactivation and recovery ratio

    inactivation rate: K_fi/Ka
    recovery rate:     K_fr/Ka

    Output
    ------
    [K_fi/Ka, K_fr/Ka]
        list of tuples of two rates. 2 tuples for 2 pathway cells
    '''
    if pathway == 1:
        thetaK = theta[10:17]
        return [thetaK[1]/thetaK[0], thetaK[2]/thetaK[0]]

    else:
        thetaK_on = theta[10:17]
        thetaK_off = theta[28:35]
        return [(thetaK_on[1]/thetaK_on[0], thetaK_on[2]/thetaK_on[0]), (thetaK_off[1]/thetaK_off[0], thetaK_off[2]/thetaK_off[0])]

def get_nonlinearity_param(theta, pathway):
    '''
    get nonlinearity parameters

    Formula
    -------
        a * (x - thr), where a is slope and thr is threshold of the nonlinearity

    Output
    ------
    [(slope, threshold)]
        list of tuples of slope and threshold. 2 tuples for 2 pathway cells
    '''

    if pathway == 1:
        thetaN = theta[8:10]
        return [thetaN[1], thetaN[0]/thetaN[1]]
    else:
        thetaN_on = theta[8:10]
        thetaN_off = theta[26:28]
        return [(thetaN_on[1], thetaN_on[0]/thetaN_on[1]), (thetaN_off[1], thetaN_off[0]/thetaN_off[1])]

