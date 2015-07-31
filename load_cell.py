import scipy.io
import numpy as np
from os import listdir
from os.path import join
import pdb

path = 'data'
class cell(object):
    def __init__(self, cell_name):
        self.cell_name = cell_name

        data = scipy.io.loadmat(join(path,cell_name))

        self.stim   = data['data'][:, 2].tolist()
        self.mp     = data['data'][:, 0].tolist()
        self.resp   = data['data'][:, 1].tolist()

    def reshape(self, time_steps, stride):
        '''
        Reshape stim, resp and mp to be each a 2D ndarray of shape (samples, time_steps)
        where samples = (len(stim) - time_steps) / stride
        '''

        start = list(range(0, len(self.stim)-time_steps, stride))
        end = [s + time_steps for s in start]

        def reshape_data(start, end, data):
            nb_samples = len(start)
            aa = map(lambda s,e, list_in : list_in[s:e], start, end, [data]*nb_samples)
            return aa

        pdb.set_trace()
        self.mp_2D   = np.array(reshape_data(start, end, self.mp))
        self.stim_2D = np.array(reshape_data(start, end, self.stim))
        self.resp_2D = np.array(reshape_data(start, end, self.resp))
        

