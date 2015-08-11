#Deep Convolutional Neural Network for predicting the membrane potential of cells from their stimulus and spikes
# Layer-by-Layer Filter Architecture:
	# nb_filters1 = 10
	# f_size1 = 400 #larger filters get more general contexts and occur at beginning layers
	# nb_filters2 = 20
	# f_size2 = 300
	# nb_filters3 = 30
	# f_size3 = 200
	# nb_filters4 = 40
	# f_size4 = 100 #smallest filters toward the end for more fine-grained context

import matplotlib
#Force matplotlib to not use any Xwindows
matplotlib.use('Agg')
import sys
sys.path.append('/afs/ir.stanford.edu/users/a/n/anayebi/LNKS/')
import loaddatatools as ldt
import numpy as np
import pickle
from keras.models import Graph
from keras.layers.core import Activation, Reshape, Permute, Dense2D
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, UpSample1D, UpSample2D
from keras.regularizers import l1, l2, l1l2
from keras.callbacks import Callback
from keras.optimizers import RMSprop, SGD
import theano.tensor as T
import matplotlib.pyplot as plt

num_epochs = 1000
rec_length = 150 #in seconds, max value is 300
rec_length = 1000*rec_length

class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
	
	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))

def loadData(cell_define):
	cells = ldt.loadcells()
	cell = cells[cell_define]
	# makes this rec_length by 2
	X_train = np.hstack((cell.st[:rec_length, np.newaxis], cell.fr[:rec_length, np.newaxis]))
	# makes this 1 by rec_length by 2
	X_train = X_train[np.newaxis, :]
	y_train = cell.mp[np.newaxis, :rec_length, np.newaxis]
	return X_train, y_train

def model1D(X_train, y_train, X_test, num_layers):
	graph = Graph()
	graph.add_input(name='stim-fr', ndim=3)
	bmode = 'same'
	init = 'uniform'
	input_dim = 2
	reg = 0.0
	input = 'stim-fr'
	out_arr = []
	label = 1

	for i in xrange(num_layers):
		nb_filters = (i+1)*10
		f_size = (num_layers - i)*100
		name = 'conv'+str(label)
		graph.add_node(Convolution1D(input_dim, nb_filters, f_size, init=init, activation='relu', border_mode=bmode, W_regularizer=l1l2(reg)),
						name=name, input=input)
		if nb_filters>1:
			label += 1
			name = 'conv'+str(label)
			input_name = 'conv'+str(label-1)
			graph.add_node(Convolution1D(nb_filters, 1, 1, init=init, activation='relu', border_mode=bmode),
				   name=name, input=input_name)
			out = name
		else:
			out = name

		name = 'pool'+str(label)
		graph.add_node(MaxPooling1D(pool_length=2), name=name, input=out)
		out = name
		name = 'upsample'+str(label)
		num_upsample = int(np.exp2(i+1))
		graph.add_node(UpSample1D(upsample_length=num_upsample), name=name, input=out)
		#updates for the next iteration of the loop
		label += 1 
		input_dim = 1
		input = out
		out_arr.append(name)

	graph.add_node(Dense2D(rec_length, num_layers, 1, W_regularizer=l1l2(reg)), 
		name='weighted_sum', inputs=out_arr, merge_mode='concat')

	graph.add_output(name='voltages', input='weighted_sum')
	graph.compile('rmsprop', {'voltages':'mse'})
	history = LossHistory()
	graph.fit({'stim-fr':X_train, 'voltages':y_train}, nb_epoch=num_epochs, verbose = 1, callbacks=[history])
	graph.save_weights("weights" + str(num_epochs)+".hdf5", overwrite=True)
	predictions = graph.predict({'stim-fr':X_test})
	predictions = predictions['voltages'][0] #since we only have 1 training example
	print predictions.shape

	#Figure to visualize predictions
	plt.plot(predictions)
	filename = '%dpredictions.png' %(num_epochs)
	plt.savefig(filename)
	plt.close()
	pickle.dump(predictions, open("predictions"+str(num_epochs)+".p", "wb"))
	
	#Figure to visualize loss history after each batch
	fig = plt.gcf()
	fig.set_size_inches((20,24))
	ax = plt.subplot()
	ax.plot(history.losses, 'k')
	ax.set_title('Loss history', fontsize=16)
	ax.set_xlabel('Iteration', fontsize=16)
	ax.set_ylabel('Loss', fontsize=14)
	plt.tight_layout()
	filename = '%dEpochs.png' %(num_epochs)
	plt.savefig(filename, bbox_inches='tight')
	plt.close()

	pickle.dump(history.losses, open("losshistory"+str(num_epochs)+".p", "wb"))

cell_define = 'g11'
[X_train, y_train] = loadData(cell_define)
print X_train.shape
print y_train.shape
X_test = X_train
num_layers = 4
model1D(X_train, y_train, X_test, num_layers)
