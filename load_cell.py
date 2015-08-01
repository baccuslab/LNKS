import scipy.io
import numpy as np
from os import listdir
from os.path import join
import sys
import pdb

path = 'data'
class Cell(object):
    def __init__(self, cell_name, time_steps):
        #if not sys.version_info < (3,):
        #    raise ValueError('This code is designed to run in python 2.7')

        self.cell_name = cell_name
        self.time_steps = time_steps

        data = scipy.io.loadmat(join(path,cell_name))

        self.stim   = data['data'][:, 2].tolist()
        self.mp     = data['data'][:, 0].tolist()
        self.spikes   = data['data'][:, 1].tolist()

    def reshape(self, stride=1):
        '''
        Reshape stim, spikes and mp to be each a 2D ndarray of shape (samples, self.time_steps)
        where samples = (len(stim) - self.time_steps) / stride
        '''

        start = list(range(0, len(self.stim)-self.time_steps, stride))
        end = [s + self.time_steps for s in start]

        def reshape_data(start, end, data):
            nb_samples = len(start)
            aa = map(lambda s,e, list_in : list_in[s:e], start, end, [data]*nb_samples) 
            if sys.version_info > (3,):
                return list(aa)
            else:
                return aa

        self.mp_2D   = np.array(reshape_data(start, end, self.mp)).reshape(-1, self.time_steps, 1)
        self.stim_2D = np.array(reshape_data(start, end, self.stim)).reshape(-1, self.time_steps, 1)
        self.spikes_2D = np.array(reshape_data(start, end, self.spikes)).reshape(-1, self.time_steps, 1)
        
    def get_dict(self):
        if not hasattr(self, 'stim_2D'):
            self.reshape()

        self.cell_dict = {}
        self.cell_dict['stim'] = self.stim_2D
        self.cell_dict['mp'] = self.mp_2D
        self.cell_dict['spikes'] = self.spikes_2D

        return self.cell_dict
