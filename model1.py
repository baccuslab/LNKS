from load_cell import Cell
from keras.models import Graph
from keras.layers.core import Activation, Reshape
from keras.activations import linear, relu
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD

class model():
    def __init__(self):
        graph = Graph()
        graph.add_input(name='stim', ndim=2)
        #graph.add_input(name='spikes', ndim=3)

        """
        graph.add_node(Reshape(100, 1),
                name='stim_3D', input='stim')
        graph.add_node(Reshape(100, 1),
                name='spikes_3D', input='spikes')
        """

        #graph.add_node(Activation(linear), 
        #        name='concat', inputs=['stim', 'spikes'], merge_mode='concat')
        graph.add_node(Activation(linear), 
                name='concat', input='stim')

        graph.add_node(LSTM(1, 1, return_sequences=True),
                name='LSTM', input='concat')

        graph.add_output(name='mp', input='LSTM')
        
        graph.compile('rmsprop', {'mp':'mse'})

        self.graph = graph
