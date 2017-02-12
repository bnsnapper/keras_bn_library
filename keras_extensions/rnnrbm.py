from __future__ import absolute_import
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
from keras.layers.recurrent import Recurrent, SimpleRNN, time_distributed_dense
from keras_extensions.initializations import glorot_uniform_sigm
from keras_extensions.activations import nrlu
from keras_extensions.rbm import RBM
from keras_extensions.dbn import DBN

class RNNRBM(Recurrent):
	def __init__(self, hidden_dim, hidden_recurrent_dim,
				init='glorot_uniform', inner_init='orthogonal',
				activation='tanh',
                W_regularizer=None, U_regularizer=None, b_regularizer=None,
				dropout_W=0., dropout_U=0.,
				nb_gibbs_steps=1,
				persistent=False,
				finetune=False,
				Wrbm_regularizer=None,
				rbm=None,
				dropout_RBM=0.,
				**kwargs):

		self.init = initializations.get(init)
		self.init_rbm = glorot_uniform_sigm
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.U_regularizer = regularizers.get(U_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.dropout_W, self.dropout_U = dropout_W, dropout_U
		self.dropout_RBM = dropout_RBM
		self.Wrbm_regularizer = regularizers.get(Wrbm_regularizer)
		self.rbm = rbm

		if self.dropout_W or self.dropout_U or self.dropout_RBM:
			self.uses_learning_phase = True 
			self.supports_masking = True

		super(RNNRBM, self).__init__(**kwargs)

		self.finetune = finetune
		self.hidden_dim = hidden_dim
		self.hidden_recurrent_dim = hidden_recurrent_dim
		self.nb_gibbs_steps = nb_gibbs_steps
		self.persistent = persistent

	def get_output_shape_for(self, input_shape):
		#assert input_shape and len(input_shape) == 2
		return (input_shape[0], self.output_dim)

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		input_dim = input_shape[2]
		self.input_dim = input_dim

		if self.stateful:
			self.reset_states()
		else:
			self.states = [None, None, None]
			self.states_dim = [self.hidden_recurrent_dim, self.input_dim, self.hidden_dim]

		if(not self.finetune):
			self.output_dim = self.input_dim
		else:
			self.output_dim = self.hidden_dim

		if(not hasattr(self, 'W')):
			self.W = self.add_weight((input_dim, self.hidden_recurrent_dim),
									initializer=self.init,
									name='{}_W'.format(self.name),
									regularizer=self.W_regularizer)
			self.U = self.add_weight((self.hidden_recurrent_dim, self.hidden_recurrent_dim),
									initializer=self.inner_init,
									name='{}_U'.format(self.name),
									regularizer=self.U_regularizer)
			self.b = self.add_weight((self.hidden_recurrent_dim,),
									initializer='zero',
									name='{}_b'.format(self.name),
									regularizer=self.b_regularizer)

			if self.initial_weights is not None:
				self.set_weights(self.initial_weights)
				del self.initial_weights

			if(self.rbm):
				self.Wrbm = self.rbm.Wrbm
				self.bv = self.rbm.bx
				self.bh = self.rbm.bh
			else:
				self.Wrbm = self.add_weight((input_dim, self.hidden_dim),
										initializer=self.init_rbm,
										name='{}_Wrbm'.format(self.name),
										regularizer=self.Wrbm_regularizer)
				self.bv = self.add_weight((self.input_dim,),
										initializer='zero',
										name='{}_bv'.format(self.name),
										regularizer=None)
				self.bh = self.add_weight((self.hidden_dim,),
										initializer='zero',
										name='{}_bh'.format(self.name),
										regularizer=None)

			self.Wuv = self.add_weight((self.hidden_recurrent_dim, input_dim),
									initializer=self.init,
									name='{}_Wuv'.format(self.name),
									regularizer=None)
			self.Wuh = self.add_weight((self.hidden_recurrent_dim, self.hidden_dim),
									initializer=self.init,
									name='{}_Wuh'.format(self.name),
									regularizer=None)

		self.trainable_weights = [self.W, self.U, self.b, self.Wrbm, self.Wuh, self.bh]

		if(not self.finetune):
			self.trainable_weights.append(self.Wuv)
			self.trainable_weights.append(self.bv)

		self.built = True

	
	def reset_states(self):
		assert self.stateful, 'Layer must be stateful.'
		input_shape = self.input_spec[0].shape

		if not input_shape[0]:
			raise Exception('If a RNN is stateful, a complete ' +
							'input_shape must be provided (including batch size).')

		if hasattr(self, 'states'):
			K.set_value(self.states[0],
			            np.zeros((input_shape[0], self.hidden_recurrent_dim)))
			K.set_value(self.states[1],
			            np.zeros((input_shape[0], self.input_dim)))
			K.set_value(self.states[2],
			            np.zeros((input_shape[0], self.hidden_dim)))
		else:
			self.states = [K.zeros((input_shape[0], self.hidden_recurrent_dim)),
							K.zeros((input_shape[0], self.input_dim)),
							K.zeros((input_shape[0], self.hidden_dim))]


	def preprocess_input(self, x):
		if self.consume_less == 'cpu':
			input_shape = K.int_shape(x)
			input_dim = input_shape[2]
			timesteps = input_shape[1]
			return time_distributed_dense(x, self.W, self.b, self.dropout_W,
			                              input_dim, self.hidden_recurrent_dim,
			                              timesteps)
		else:
			return x

	def step(self, x, states):
		u_tm1 = states[0]
		B_U = states[3]
		B_W = states[4]

		bv_t = self.bv + K.dot(u_tm1, self.Wuv)
		bh_t = self.bh + K.dot(u_tm1, self.Wuh)

		if self.consume_less == 'cpu':
			h = x
		else:
			h = self.b + K.dot(x * B_W, self.W)

		u_t = self.activation(h + K.dot(u_tm1 * B_U, self.U))

		return x, [u_t, bv_t, bh_t]

	
	def get_constants(self, x):
		constants = []
		if 0 < self.dropout_U < 1:
			ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, self.hidden_recurrent_dim))
			B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
			constants.append(B_U)
		else:
			constants.append(K.cast_to_floatx(1.))
        
		if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
			input_shape = self.input_spec[0].shape
			input_dim = input_shape[-1]
			ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, input_dim))
			B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
			constants.append(B_W)
		else:
			constants.append(K.cast_to_floatx(1.))

		return constants

	def get_initial_states(self, x):
		print("initial state building")
		# build an all-zero tensor of shape (samples, output_dim)
		initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
		initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
		initial_state = K.expand_dims(initial_state)  # (samples, 1)
		initial_states=[]
		for dim in self.states_dim:
			initial_states.append(K.tile(initial_state, [1, dim]))  # (samples, output_dim)
		#initial_states = [initial_state for _ in range(len(self.states))]
		return initial_states

	def call(self, x, mask=None):

		input_shape = self.input_spec[0].shape

		if self.unroll and input_shape[1] is None:
			raise ValueError('Cannot unroll a RNN if the '
		                 'time dimension is undefined. \n'
		                 '- If using a Sequential model, '
		                 'specify the time dimension by passing '
		                 'an `input_shape` or `batch_input_shape` '
		                 'argument to your first layer. If your '
		                 'first layer is an Embedding, you can '
		                 'also use the `input_length` argument.\n'
		                 '- If using the functional API, specify '
		                 'the time dimension by passing a `shape` '
		                 'or `batch_shape` argument to your Input layer.')

		if self.stateful:
			initial_states = self.states
		else:
			initial_states = self.get_initial_states(x)

		constants = self.get_constants(x)
		preprocessed_input = self.preprocess_input(x)

		last_output, outputs, states = K.rnn(self.step, preprocessed_input,
		                                     initial_states,
		                                     go_backwards=self.go_backwards,
		                                     mask=mask,
		                                     constants=constants,
		                                     unroll=self.unroll,
		                                     input_length=input_shape[1])

		if self.stateful:
			updates = []
			for i in range(len(states)):
				updates.append((self.states[i], states[i]))

		u_t = states[0]
		bv_t = states[1]
		bh_t = states[2]

		if(not self.finetune):
			self.rbm_rnn = RBM(self.hidden_dim,init=glorot_uniform_sigm,
								input_dim=self.input_dim,
								hidden_unit_type='binary',
								visible_unit_type='gaussian',
								persistent=self.persistent, 
								batch_size=self.batch_input_shape[0],
								nb_gibbs_steps=self.nb_gibbs_steps, 
								Wrbm=self.Wrbm, bx=bv_t, bh=bh_t,
								dropout=self.dropout_RBM)
			self.rbm_rnn.build([input_shape[0], self.input_dim])

			self.loss = self.rbm_rnn.contrastive_divergence_loss
			self.metrics = self.rbm_rnn.reconstruction_loss

		x = K.reshape(x, (-1, self.input_dim))
		
		if(not self.finetune):
			return x
		else:
			#return K.sigmoid(K.dot(x, self.Wrbm) + bh_t)
			return K.dot(x, self.Wrbm) + bh_t
			#return last_output

	def set_finetune(self):
		self.finetune = True
		self.built = False
		self.inbound_nodes = []

	def get_config(self):
		config = {
			'hidden_dim': self.hidden_dim,
			'hidden_recurrent_dim': self.hidden_recurrent_dim,
			'init': self.init.__name__,
			'inner_init': self.inner_init.__name__,
			'activation': self.activation.__name__,
			'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
			'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
			'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
			'W_regularizer': self.Wrbm_regularizer.get_config() if self.Wrbm_regularizer else None,
			'dropout_W': self.dropout_W,
			'dropout_U': self.dropout_U,
			'dropout_RBM': self.dropout_RBM,
			'nb_gibbs_steps' : self.nb_gibbs_steps, 
			'persistent' : self.persistent,
			'finetune' : self.finetune
		}
		base_config = super(RNNRBM, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

########################################################################
#
#
#
#                               RNN-DBN
#
#
#
########################################################################
