'''
Models to predict membrane potential from spikes and stimulus
'''
from code.load_cell import Cell
from keras.models import Graph
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D
from keras.activations import linear, relu
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l1, l2, l1l2
import theano.tensor as T
from matplotlib import pyplot as plt

class model1(object):
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
        
        rmsprop = RMSprop(lr=1E-10)
        graph.compile(rmsprop, {'mp':'mse'})

        self.graph = graph

class model2D(object):
    '''
    Stimulus and Spikes are treated as very long images. 
    Implementing Convolution2D
    Keras expects inputs to Convolution1D to be (nb_samples, steps, input_dim)
    However I could split stim/spikes into shorter sequences and call each one of them a sample.
    '''
    def __init__(self, f_size, nb_filters, reg=0.01, lr=0.01):
        self.f_size = f_size
        self.nb_filters = nb_filters

        graph = Graph()

        graph.add_input(name='stim', ndim=4)

        nl = 'softplus'

        graph.add_node(Convolution2D(nb_filters, 1, f_size, 1, activation=nl, border_mode='same', W_regularizer=l1l2(reg)),
                name='conv1', input='stim')

        if nb_filters>1:
            graph.add_node(Convolution2D(1, nb_filters, 1, 1, activation=nl, border_mode='valid'),
                   name='conv2', input='conv1')

            out = 'conv2'
        else:
            out = 'conv1'

        graph.add_output(name='spikes', input=out)

        rmsprop = RMSprop(lr=lr)
        graph.compile(rmsprop, {'spikes':poisson_loss})

        self.graph = graph
        

    def fit(self, cell, nb_epoch=20):
        if cell.stim.ndim != 4:
            cell.reshape4D()

        self.graph.fit({'stim':cell.stim, 'spikes':cell.spikes},
                nb_epoch=nb_epoch)

    def show_filters(self):
        filters = self.graph.params[0].eval()
        if len(filters)>1:
            fig, ax = plt.subplots(num='filters', nrows=2)

            for f in filters:
                ax[0].plot(f.flatten())


            ax[1].plot(self.graph.params[2].eval()[0,:,0,0])
        else:
            plt.plot(filters[0,0,:,0])

        plt.show()

class model1D(model2D):
    '''
    Stimulus and Spikes are treated as very long images. 
    Implementing Convolution1D
    Keras expects inputs to Convolution1D to be (nb_samples, steps, input_dim)
    However I could split stim/spikes into shorter sequences and call each one of them a sample.
    '''
    def __init__(self, f_size, nb_filters, reg=0.01, lr=0.01):
        self.f_size = f_size
        self.nb_filters = nb_filters

        graph = Graph()

        graph.add_input(name='stim', ndim=3)

        nl = 'softplus'

        graph.add_node(Convolution1D(1, nb_filters, f_size,
            activation=nl, border_mode='valid', W_regularizer=l1l2(reg)),
                name='conv1', input='stim')

        if nb_filters>1:
            graph.add_node(Convolution1D(nb_filters, 1, 1,
                activation=nl, border_mode='valid'),
                   name='conv2', input='conv1')

            out = 'conv2'
        else:
            out = 'conv1'

        graph.add_output(name='spikes', input=out)

        rmsprop = RMSprop(lr=lr)
        graph.compile(rmsprop, {'spikes':poisson_loss})

        self.graph = graph

    def fit(self, cell, nb_epoch=20):
        if cell.stim.ndim != 3:
            cell.reshape3D()

        self.graph.fit({'stim':cell.stim, 'spikes':cell.spikes[:,:-(self.f_size-1),:]}
                , nb_epoch=nb_epoch)

def poisson_loss(y_true, y_pred):
    return T.mean(y_pred - y_true * T.log(y_pred), axis=-1)
