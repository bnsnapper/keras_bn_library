import numpy as np

from keras_extensions.rbm import RBM
from keras_extensions.initializations import glorot_uniform_sigm
from keras_extensions.layers import SampleBernoulli
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
#from keras_extensions.models import SingleLayerUnsupervised

import keras.backend as K

class DBN(object):

	def __init__(self, rbms):
		self.rbms = rbms

		# biases
		self.bs = []
		# weights
		self.Ws  = []

		# building the list of DBN parameters
		self.bs.append(self.rbms[0].bx)
		for rbm in self.rbms:
			self.bs.append(rbm.bh)
			self.Ws.append(rbm.Wrbm)

        # DBN params are in bs and Ws
        
		for indx,rbm in enumerate(self.rbms):
			if indx != 0:
				rbm.bx = self.bs[indx]
			rbm.bh = self.bs[indx+1]

			if indx != 0:
				rbm.trainable_weights = [self.Ws[indx],  self.bs[indx+1]]
			else:
				rbm.trainable_weights = [self.Ws[indx], self.bs[indx],  self.bs[indx+1]]

	def compile(self, layer_optimizer, layer_loss, metrics=[]):
		self.get_layer_optimizer = layer_optimizer
		self.get_layer_loss      = layer_loss
		self.metrics = metrics

	def fit(self, X, batch_size=128, nb_epochs=100, verbose=1, shuffle="batch",
            callbacks=[]):

		self.X = X
		self.batch_size= batch_size

		if(type(nb_epochs) != list):
			self.nb_epochs = [nb_epochs for i in range(len(self.rbms))]
		else:
			self.nb_epochs = nb_epochs

		self.verbose = verbose
		self.shuffle = shuffle

		for i in range(len(self.rbms)):

			if(len(callbacks) == 0):
				cbk = []
			else:
				cbk = callbacks[i]

			if(len(self.metrics) == 0):
				mtrs = []
			else:
				mtrs = self.metrics[i]

			self.greedy_layerwise_train(layer_no=i, 
				nb_epochs=self.nb_epochs[i],
				callbacks=cbk,
				metrics=mtrs)

	def greedy_layerwise_train(self, layer_no, nb_epochs, callbacks=[],
				metrics=[]):

		print (">>>", layer_no)

		# preparing input for each layer
		if(layer_no == 0):
			ins = self.X
		else:
			pre_model = Sequential()
           
			for i,rbm in enumerate(self.rbms[0:layer_no]):
				pre_model.add(rbm.get_h_given_x_layer((i==0)))
		      
				if(rbm.hidden_unit_type == 'nrlu'):
					pre_model.add(SampleBernoulli(mode='nrlu'))
				else:
					pre_model.add(SampleBernoulli(mode='random'))

			pre_model.compile(SGD(),loss='mean_squared_error')
			ins = pre_model.predict(self.X)

		input_dim = self.rbms[layer_no].input_dim
		# preparing model
		#model = SingleLayerUnsupervised()
		model = Sequential()
		model.add(self.rbms[layer_no])

		loss = self.get_layer_loss(self.rbms[layer_no], layer_no)
		opt  = self.get_layer_optimizer(layer_no)

		model.compile(optimizer=opt, loss=loss, metrics=metrics)
		model.summary()

		model.fit(
            ins,ins,
            batch_size = self.batch_size,
            nb_epoch = nb_epochs,
            verbose= self.verbose,
            shuffle = self.shuffle,
		    callbacks = callbacks
        )

	def get_forward_inference_layers(self):
		L = []
		for ind in range(len(self.rbms)):
			L.append(
				self.rbms[ind].get_h_given_x_layer((ind==0))
			)
		return L

	def get_backward_inference_layers(self):
		L = []
		for ind in reversed(range(len(self.rbms))):
			L.append(
				self.rbms[ind].get_x_given_h_layer()
		)
		return L
