'''
Models to predict membrane potential from spikes and stimulus
'''
from load_cell import Cell
from keras.models import Graph
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution1D
from keras.activations import linear, relu
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import RMSprop, SGD

class model1():
    def __init__(self):
        graph = Graph()
        graph.add_input(name='stim', ndim=3)
        graph.add_input(name='spikes', ndim=3)

        """
        graph.add_node(Reshape(100, 1),
                name='stim_3D', input='stim')
        graph.add_node(Reshape(100, 1),
                name='spikes_3D', input='spikes')
        """

        graph.add_node(Activation(linear), 
                name='concat', inputs=['stim', 'spikes'], merge_mode='concat')
        
        graph.add_node(SimpleRNN(2, 1, return_sequences=True),
                name='SimpleRNN', input='concat')

        graph.add_output(name='mp', input='SimpleRNN')
        
        graph.compile('rmsprop', {'mp':'mse'})

        self.graph = graph

class model2():
    '''
    Stimulus and Spikes are treated as very long images. I'll do either Convolution1D or Convolution2D
    Keras expects inputs to Convolution1D to be (nb_samples, steps, input_dim)
    The way I'm thinking about it both spikes and stim are just 1 sample with many many steps and one input_dim
    However I could split stim/spikes into shorter sequences and call each one of them a sample.
    '''
    def __init__(self):
        graph = Graph()
        graph.add_input(name='stim', ndim=3)
        #graph.add_input(name='spikes', ndim=3)

        graph.add_node(Convolution1D(1, 1, 300, activation='relu', border_mode='valid'),
                name='conv1', input='stim')

        graph.add_output(name='spikes', input='conv1')

        graph.compile('rmsprop', {'spikes':'mse'})

        self.graph = graph
        
