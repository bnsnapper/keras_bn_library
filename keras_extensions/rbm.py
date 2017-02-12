from __future__ import division

import numpy as np

import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import activations, initializations, regularizers, constraints
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.layers.core import Dense, Flatten
from keras_extensions.initializations import glorot_uniform_sigm
from keras_extensions.activations import nrlu

class RBM(Layer):

	def __init__(self, hidden_dim, init='glorot_uniform', 
		weights=None, 
		Wrbm_regularizer=None, bx_regularizer=None, bh_regularizer=None, 
		activity_regularizer=None,
		Wrbm_constraint=None, bx_constraint=None, bh_constraint=None,
		input_dim=None, nb_gibbs_steps=1, persistent=False, batch_size=1,
		scaling_h_given_x=1.0, scaling_x_given_h=1.0,
		dropout=0.0,
		hidden_unit_type='binary',
		visible_unit_type='binary',
		Wrbm=None, bh=None, bx=None,
		**kwargs):

		self.p = dropout
		if(0.0 < self.p < 1.0): 
			self.uses_learning_phase = True 
			self.supports_masking = True 

		if(hidden_unit_type == 'softmax'):
			activation = 'softmax'
			self.is_persistent = False
			self.nb_gibbs_steps = 1
		elif(hidden_unit_type == 'nrlu'):
			activation = 'relu'
			self.is_persistent = False
			self.nb_gibbs_steps = 1
		else:
			activation = 'sigmoid'
			self.is_persistent = persistent
			self.nb_gibbs_steps = nb_gibbs_steps

		self.updates = []
		self.init = initializations.get(init)
		self.activation = activations.get(activation)
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.batch_size = batch_size
		self.hidden_unit_type = hidden_unit_type
		self.visible_unit_type = visible_unit_type

		self.scaling_h_given_x = scaling_h_given_x
		self.scaling_x_given_h = scaling_x_given_h

		self.Wrbm_regularizer = regularizers.get(Wrbm_regularizer)
		self.bx_regularizer = regularizers.get(bx_regularizer)
		self.bh_regularizer = regularizers.get(bh_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.Wrbm_constraint = constraints.get(Wrbm_constraint)
		self.bx_constraint = constraints.get(bx_constraint)
		self.bh_constraint = constraints.get(bh_constraint)

		self.initial_weights = weights
		self.input_spec = [InputSpec(ndim='2+')]
	
		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		
		super(RBM, self).__init__(**kwargs)


		if(Wrbm == None):
			self.Wrbm = self.add_weight((input_dim, self.hidden_dim),
									initializer=self.init,
									name='{}_Wrbm'.format(self.name),
									regularizer=self.Wrbm_regularizer,
									constraint=self.Wrbm_constraint)
		else:
			self.Wrbm = Wrbm

		if(bx == None):
			self.bx = self.add_weight((self.input_dim,),
								initializer='zero',
								name='{}_bx'.format(self.name),
								regularizer=self.bx_regularizer,
								constraint=self.bx_constraint)
		else:
			self.bx = bx

		if(bh == None):
			self.bh = self.add_weight((self.hidden_dim,),
								initializer='zero',
								name='{}_bh'.format(self.name),
								regularizer=self.bh_regularizer,
								constraint=self.bh_constraint)
		else:
			self.bh = bh

		if(self.is_persistent):
			self.persistent_chain = K.variable(np.zeros((self.batch_size, self.input_dim), 
												dtype=K.floatx()))

	def _get_noise_shape(self, x): 
		return None 


	def build(self, input_shape):
		assert len(input_shape) == 2
		input_dim = input_shape[-1]
		self.input_spec = [InputSpec(dtype=K.floatx(),
									ndim='2+')]

		#self.trainable_weights = [self.W, self.bx, self.bh]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
		del self.initial_weights

	def call(self, x, mask=None):
		return 1.0*x

	def get_output_shape_for(self, input_shape):
		#assert input_shape and len(input_shape) == 2
		return (input_shape[0], self.input_dim)

	def get_config(self):
		config = {'output_dim': self.hidden_dim,
		'init': self.init.__name__,
		'activation': self.activation.__name__,
		'Wrbm_regularizer': self.Wrbm_regularizer.get_config() if self.Wrbm_regularizer else None,
		'bh_regularizer': self.bh_regularizer.get_config() if self.bh_regularizer else None,
		'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer  else None,
		'Wrbm_constraint': self.Wrbm_constraint.get_config() if self.Wrbm_constraint else None,
		'bh_constraint': self.bh_constraint.get_config() if self.bh_constraint else None,
		'input_dim': self.input_dim
		}

		base_config = super(Dense, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

    # -------------
    # RBM internals
    # -------------

	def free_energy(self, x):
		wx_b = K.dot(x, self.Wrbm) + self.bh

		if(self.visible_unit_type == 'gaussian'):
			vbias_term = 0.5*K.sum((x - self.bx)**2, axis=1)
			hidden_term = K.sum(K.log(1 + K.exp(wx_b)), axis=1)
			return -hidden_term + vbias_term
		else:
			hidden_term = K.sum(K.log(1 + K.exp(wx_b)), axis=1)
			vbias_term = K.dot(x, self.bx)
			return -hidden_term - vbias_term


	def sample_h_given_x(self, x):
		h_pre = K.dot(x, self.Wrbm) + self.bh	
		h_sigm = self.activation(self.scaling_h_given_x * h_pre)

		# drop out noise
		#if(0.0 < self.p < 1.0):
		#	noise_shape = self._get_noise_shape(h_sigm)
		#	h_sigm = K.in_train_phase(K.dropout(h_sigm, self.p, noise_shape), h_sigm)
		
		if(self.hidden_unit_type == 'binary'):
			h_samp = K.random_binomial(shape=h_sigm.shape, p=h_sigm)
	        # random sample
	        #   \hat{h} = 1,      if p(h=1|x) > uniform(0, 1)
	        #             0,      otherwise
		elif(self.hidden_unit_type == 'nrlu'):
			h_samp = nrlu(h_pre)
		else:
			h_samp = h_sigm

		if(0.0 < self.p < 1.0):
			noise_shape = self._get_noise_shape(h_samp)
			h_samp = K.in_train_phase(K.dropout(h_samp, self.p, noise_shape), h_samp)

		return h_samp, h_pre, h_sigm


	def sample_x_given_h(self, h):
		x_pre = K.dot(h, self.Wrbm.T) + self.bx 

		if(self.visible_unit_type == 'gaussian'):
			x_samp = self.scaling_x_given_h  * x_pre
			return x_samp, x_samp, x_samp
		else:       
			x_sigm = K.sigmoid(self.scaling_x_given_h  * x_pre)             
			x_samp = K.random_binomial(shape=x_sigm.shape, p=x_sigm)
			return x_samp, x_pre, x_sigm


	def gibbs_xhx(self, x0):
		h1, h1_pre, h1_sigm = self.sample_h_given_x(x0)
		x1, x1_pre, x1_sigm = self.sample_x_given_h(h1)
		return x1, x1_pre, x1_sigm


	def mcmc_chain(self, x, nb_gibbs_steps):
		xi = x
		for i in range(nb_gibbs_steps):
			xi, xi_pre, xi_sigm = self.gibbs_xhx(xi)
		x_rec, x_rec_pre, x_rec_sigm = xi, xi_pre, xi_sigm

		x_rec = K.stop_gradient(x_rec)

		return x_rec, x_rec_pre, x_rec_sigm


	def contrastive_divergence_loss(self, y_true, y_pred):

		x = y_pred
		#x = K.reshape(x, (-1, self.input_dim))

		if(self.is_persistent):
			chain_start = self.persistent_chain
		else:
			chain_start = x

		def loss(chain_start, x):
			x_rec, _, _ = self.mcmc_chain(chain_start, self.nb_gibbs_steps)
			cd = K.mean(self.free_energy(x)) - K.mean(self.free_energy(x_rec))
			return cd, x_rec

		y, x_rec = loss(chain_start, x)

		if(self.is_persistent):
			self.updates = [(self.persistent_chain, x_rec)]

		return y


	def reconstruction_loss(self, y_true, y_pred):

		x = y_pred

		def loss(x):
			if(self.visible_unit_type == 'gaussian'):
				x_rec, _, _ = self.mcmc_chain(x, self.nb_gibbs_steps)
				return K.mean(K.sqrt(x - x_rec))
			else:
				_, pre, _ = self.mcmc_chain(x, self.nb_gibbs_steps)
				cross_entropy_loss = -K.mean(K.sum(x*K.log(K.sigmoid(pre)) + 
										(1 - x)*K.log(1 - K.sigmoid(pre)), axis=1))
				return cross_entropy_loss

		return loss(x)


	def free_energy_gap(self, x_train, x_test):
		return K.mean(self.free_energy(x_train)) - K.mean(self.free_energy(x_test))


	def get_h_given_x_layer(self, as_initial_layer=False):

		if(as_initial_layer):
			layer = Dense(input_dim=self.input_dim, output_dim=self.hidden_dim,
							activation=self.activation, 
							weights=[self.Wrbm.get_value(), self.bh.get_value()])
		else:
			layer = Dense(output_dim=self.hidden_dim, 
							activation=self.activation, 
							weights=[self.Wrbm.get_value(), self.bh.get_value()])
		return layer


	def get_x_given_h_layer(self, as_initial_layer=False):

		if(self.visible_unit_type == 'gaussian'):
			act = 'linear'
		else:
			act = 'sigmoid'

		if(as_initial_layer):
			layer = Dense(input_dim=self.hidden_dim, output_dim=self.input_dim, 
						activation=act, 
						weights=[self.Wrbm.get_value().T, self.bx.get_value()])
		else:
			layer = Dense(output_dim=self.input_dim, activation=act, 
						weights=[self.Wrbm.get_value().T, self.bx.get_value()])
		return layer


	def return_reconstruction_data(self, x):

		def re_sample(x):
			x_rec, pre, _ = self.mcmc_chain(x, self.nb_gibbs_steps)
			return x_rec

		return re_sample(x)

