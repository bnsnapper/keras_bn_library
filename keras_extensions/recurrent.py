# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

import keras.backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers.recurrent import Recurrent, time_distributed_dense
from keras.layers.core import Flatten

class DecoderVaeLSTM(Recurrent):
	def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):

		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.U_regularizer = regularizers.get(U_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.dropout_W, self.dropout_U = dropout_W, dropout_U

		if self.dropout_W or self.dropout_U:
			self.uses_learning_phase = True
		super(DecoderVaeLSTM, self).__init__(**kwargs)

	def get_initial_states(self, x):
		print("initial state building")
		# build an all-zero tensor of shape (samples, output_dim)
		initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
		initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
		initial_state = K.expand_dims(initial_state)  # (samples, 1)
		initial_state = K.tile(initial_state, [1, self.input_dim])

		initial_states = [initial_state for _ in range(len(self.states))]
		return initial_states


	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		self.input_dim = input_shape[2]

		self.W = self.init((self.output_dim, 4 * self.input_dim),
		                   name='{}_W'.format(self.name))
		self.U = self.inner_init((self.input_dim, 4 * self.input_dim),
		                         name='{}_U'.format(self.name))
		self.b = K.variable(np.hstack((np.zeros(self.input_dim),
		                               K.get_value(self.forget_bias_init((self.input_dim,))),
		                               np.zeros(self.input_dim),
		                               np.zeros(self.input_dim))),
		                    name='{}_b'.format(self.name))

		self.A = self.init((self.input_dim, self.output_dim),
		                    name='{}_A'.format(self.name))
		self.ba = K.zeros((self.output_dim,), name='{}_ba'.format(self.name))


		self.trainable_weights = [self.W, self.U, self.b, self.A, self.ba]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights


	def step(self, x, states):
		h_tm1 = states[0]
		c_tm1 = states[1]

		x_t = self.activation(K.dot(h_tm1, self.A) + self.ba)
		z = K.dot(x_t, self.W) + K.dot(h_tm1, self.U) + self.b


		z0 = z[:, :self.input_dim]
		z1 = z[:, self.input_dim: 2 * self.input_dim]
		z2 = z[:, 2 * self.input_dim: 3 * self.input_dim]
		z3 = z[:, 3 * self.input_dim:]

		i = self.inner_activation(z0)
		f = self.inner_activation(z1)
		c = f * c_tm1 + i * self.activation(z2)
		o = self.inner_activation(z3)

		h = o * self.activation(c)

		return x_t, [h, c]



	def call(self, x, mask=None):

		input_shape = self.input_spec[0].shape

		# state format: [h(t-1), c(t-1), y(t-1)]
		#h_0 = K.zeros_like(x[:, 0, :])
		#c_0 = K.zeros_like(x[:, 0, :])
		h_0 = K.reshape(x, (-1, self.input_dim))
		c_0 = K.reshape(x, (-1, self.input_dim))
		initial_states = [h_0, c_0]

		#self.states = [None, None]
		#initial_states = self.get_initial_states(x)

		last_output, outputs, states = K.rnn(step_function=self.step, 
                                             inputs=x,
                                             initial_states=initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=None,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])

		if self.return_sequences:
			return outputs
		else:
			return last_output


	def get_config(self):
		config = {'output_dim': self.output_dim,
		          'init': self.init.__name__,
		          'inner_init': self.inner_init.__name__,
		          'forget_bias_init': self.forget_bias_init.__name__,
		          'activation': self.activation.__name__,
		          'out_activation': self.out_activation.__name__,
		          'inner_activation': self.inner_activation.__name__}
		base_config = super(DecoderVaeLSTM, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))






class QRNN(Recurrent):

	def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh', inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.U_regularizer = regularizers.get(U_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.dropout_W = dropout_W
		self.dropout_U = dropout_U
		self.stateful = False

		if self.dropout_W or self.dropout_U:
			self.uses_learning_phase = True
		super(QRNN, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		input_dim = input_shape[2]
		self.input_dim = input_dim
		
		if self.stateful:
			self.reset_states()
		else:
			self.states = [None, None]
			self.states_dim = [self.input_dim, self.output_dim]


		self.weight_size = self.output_dim * 4
		self.W = self.add_weight((input_dim, self.weight_size),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)
		self.U = self.add_weight((input_dim, self.weight_size),
                                 initializer=self.inner_init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer)

		def b_reg(shape, name=None):
			return K.variable(np.hstack((np.zeros(self.output_dim),
										K.get_value(self.forget_bias_init((self.output_dim,))),
										np.zeros(self.output_dim),
										np.zeros(self.output_dim))),
										name='{}_b'.format(self.name))
		self.b = self.add_weight((self.weight_size,),
                                     initializer=b_reg,
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer)


		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

		self.built = True

	def reset_states(self):
		assert self.stateful, 'Layer must be stateful.'
		input_shape = self.input_spec[0].shape
		if not input_shape[0]:
			raise ValueError('If a RNN is stateful, it needs to know '
			                 'its batch size. Specify the batch size '
			                 'of your input tensors: \n'
			                 '- If using a Sequential model, '
			                 'specify the batch size by passing '
			                 'a `batch_input_shape` '
			                 'argument to your first layer.\n'
			                 '- If using the functional API, specify '
			                 'the time dimension by passing a '
			                 '`batch_shape` argument to your Input layer.')
		if hasattr(self, 'states'):
			K.set_value(self.states[0],
			            np.zeros((input_shape[0], self.input_dim)))
			K.set_value(self.states[1],
			            np.zeros((input_shape[0], self.output_dim)))
		else:
			self.states = [K.zeros((input_shape[0], self.input_dim)),
							K.zeros((input_shape[0], self.output_dim))]

	def get_initial_states(self, x):
		initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
		initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
		initial_state = K.expand_dims(initial_state)  # (samples, 1)
		initial_states=[]
		for dim in self.states_dim:
			initial_states.append(K.tile(initial_state, [1, dim]))
		return initial_states

	def preprocess_input(self, x):
		return x

	def step(self, x, states):
		_previous = states[0]
		_p_c = states[1]
		#B_U = states[2]
		#B_W = states[3]

		_current = K.dot(x, self.W)
		_p = K.dot(_previous, self.U) + self.b
		_weighted = _current + _p

		z0 = _weighted[:, :self.output_dim]
		z1 = _weighted[:, self.output_dim: 2 * self.output_dim]
		#z2 = _weighted[:, 2 * self.output_dim:]
		z2 = _weighted[:, 2 * self.output_dim:3 * self.output_dim]
		z3 = _weighted[:, 3 * self.output_dim:]

		i = self.inner_activation(z0)
		f = self.inner_activation(z1)
		z = self.activation(z2)
		o = self.inner_activation(z3)
		#f = self.inner_activation(z0)
		#z = self.activation(z1)
		#o = self.inner_activation(z2)
	

		c = f * _p_c + i * z
		#c = f * _p_c + (1 - f) * z
		h = self.activation(c) * o   # h is size vector
		return h, [x, c]

	def get_constants(self, x):
		constants = []
		if 0 < self.dropout_U < 1:
			ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, self.input_dim))
			B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
			constants.append(B_U)
		else:
			constants.append([K.cast_to_floatx(1.) for _ in range(4)])

		if 0 < self.dropout_W < 1:
			input_shape = K.int_shape(x)
			input_dim = input_shape[-1]
			ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, int(input_dim)))
			B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
			constants.append(B_W)
		else:
			constants.append([K.cast_to_floatx(1.) for _ in range(4)])
		return constants

	def get_config(self):
		config = {'output_dim': self.output_dim,
				'init': self.init.__name__,
				'inner_init': self.inner_init.__name__,
				'forget_bias_init': self.forget_bias_init.__name__,
				'activation': self.activation.__name__,
				'inner_activation': self.inner_activation.__name__,
				'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
				'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
				'dropout_W': self.dropout_W,
				'dropout_U': self.dropout_U}
		base_config = super(QRNN, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
