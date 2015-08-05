import scipy.io
import numpy as np
from os import listdir
from os.path import join
import sys
import pdb

path = 'data'
class Cell(object):
    def __init__(self, cell_name, time_steps=100):
        self.cell_name = cell_name
        self.time_steps = time_steps

        data = scipy.io.loadmat(join(path,cell_name))

        self.stim   = data['data'][:, 2]
        self.mp     = data['data'][:, 0]
        self.spikes   = data['data'][:, 1]


    def preprocess(self):
        self.stim -= self.stim.mean()
        self.get_spikes()
        self.reshape4D()

    def reshape(self, stride=1):
        '''
        Reshape stim, spikes and mp to be each a 2D ndarray of shape (samples, self.time_steps)
        where samples = (len(stim) - self.time_steps) / stride
        '''



        start = list(range(0, len(self.stim)-self.time_steps, stride))
        end = [s + self.time_steps for s in start]

        def reshape_data(start, end, data):
            if type(data)==np.ndarray:
                data = list(data)

            nb_samples = len(start)
            aa = map(lambda s,e, list_in : list_in[s:e], start, end, [data]*nb_samples) 
            if sys.version_info > (3,):
                return list(aa)
            else:
                return aa

        self.mp_2D   = reshape_data(start, end, self.mp).reshape(-1, self.time_steps, 1)
        self.stim_2D = reshape_data(start, end, self.stim).reshape(-1, self.time_steps, 1)
        self.spikes_2D = reshape_data(start, end, self.spikes).reshape(-1, self.time_steps, 1)
        
    def reshape3D(self, nb_samples=1):
        '''
        For Convolution1D, inputs have to be 3D tensors (nb_samples, steps, input_dim), I'm reshaping
        inputs to match this definition
        '''
        self.stim   = self.stim.reshape(1, -1, 1)
        self.spikes = self.spikes.reshape(1, -1, 1)
        self.mp     = self.spikes.reshape(1, -1, 1)

    def reshape4D(self, nb_samples=1):
        '''
        For Convolution1D, inputs have to be 3D tensors (nb_samples, steps, input_dim), I'm reshaping
        inputs to match this definition
        '''
        self.stim   = self.stim.reshape(1, 1, -1, 1)
        self.spikes = self.spikes.reshape(1, 1, -1, 1)
        self.mp     = self.spikes.reshape(1, 1,-1, 1)

    def get_dict(self):
        if not hasattr(self, 'stim_2D'):
            self.reshape()

        self.cell_dict = {}
        self.cell_dict['stim'] = self.stim_2D
        self.cell_dict['mp'] = self.mp_2D
        self.cell_dict['spikes'] = self.spikes_2D

        return self.cell_dict

    def get_spikes(self, threshold=0.8):
        '''
        change spikes to be a binary output, defining a spike whenever threshold is crossed.

        input:
        -----
            threshold:  float in (0,1)
                        defines what parts of the recording are defined as spikes (1) or no spikes (0)
                        according to:
                            
                            (spikes - spikes.min())/( spikes.max() - spikes.min() )  > threshold 

        '''
        assert threshold <1, "get_spikes requires threhsold to be more than 0 and less than 1"
        assert threshold >0, "get_spikes requires threhsold to be more than 0 and less than 1"

        self.spikes = np.where( (self.spikes - self.spikes.min())/( self.spikes.max() - self.spikes.min()) > threshold, 
                np.ones_like(self.spikes), np.zeros_like(self.spikes) )

        self.fr = self.spikes.mean()*1000    # time is in ms

    def downsample(self, n):
        '''
        downsample stim, mp, spikes by a factor of n
        '''
        pdb.set_trace()

        for k in ['spikes', 'mp', 'stim']:
            raw = getattr(self, k)

            N = len(raw)//n

            raw = raw[:N*n].reshape(n,N).mean(axis=0)
            setattr(self, k, raw)
