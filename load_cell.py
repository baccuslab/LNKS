import scipy.io
from os import listdir
from os.path import join

path = 'data'
class cell(object):
    def __init__(self, cell_name):
        self.cell_name = cell_name

        print(cell_name)
        print(join(path, cell_name))
        data = scipy.io.loadmat(join(path,cell_name))

        self.stim   = data['data'][:, 2]
        self.mp     = data['data'][:, 0]
        self.resp   = data['data'][:, 1]
