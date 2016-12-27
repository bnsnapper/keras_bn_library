# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

import keras.backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers.recurrent import Recurrent
from keras.layers.core import Flatten

class DecoderLSTM(Recurrent):
	# Since initial hidden state is replicated from the input, there should be
	# input_dim == hidden_dim

	def __init__(self, output_dim,
	             init='glorot_uniform', inner_init='orthogonal',
	             forget_bias_init='one', activation='tanh',
	             out_activation='linear', inner_activation='hard_sigmoid',
	             **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.out_activation = activations.get(out_activation)
		self.inner_activation = activations.get(inner_activation)

		super(DecoderLSTM, self).__init__(**kwargs)

	def call(self, x, mask=None):
		# input shape: (nb_samples, time (padded with zeros), input_dim)
		# note that the .build() method of subclasses MUST define
		# self.input_spec with a complete input shape.
		input_shape = self.input_spec[0].shape

		# state format: [h(t-1), c(t-1), y(t-1)]
		h_0 = K.zeros_like(x[:, 0, :])
		c_0 = K.zeros_like(x[:, 0, :])

		y_0 = K.zeros_like(x)  # (samples, timesteps, input_dim)
		y_0 = K.sum(y_0, axis=(1, 2))  # (samples,)
		y_0 = K.expand_dims(y_0)  # (samples, 1)
		y_0 = K.tile(y_0, [1, self.output_dim])  # (samples, output_dim)

		initial_states = [h_0, c_0, y_0]

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

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
		self.input_dim = input_shape[2]

		self.W = self.init((self.input_dim, 4 * self.input_dim),
		                   name='{}_W'.format(self.name))
		self.U = self.inner_init((self.input_dim, 4 * self.input_dim),
		                         name='{}_U'.format(self.name))
		self.A = self.init((self.output_dim, 4 * self.input_dim),
		                    name='{}_A'.format(self.name))
		self.b = K.variable(np.hstack((np.zeros(self.input_dim),
		                               K.get_value(self.forget_bias_init((self.input_dim,))),
		                               np.zeros(self.input_dim),
		                               np.zeros(self.input_dim))),
		                    name='{}_b'.format(self.name))
		self.V_y = self.init((self.input_dim, self.output_dim),
		                    name='{}_V_y'.format(self.name))
		self.b_y = K.zeros((self.output_dim,), name='{}_b_y'.format(self.name))

		self.trainable_weights = [self.W, self.U, self.A, self.b,
		                          self.V_y, self.b_y]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def step(self, x, states):
		h_tm1 = states[0]
		c_tm1 = states[1]
		y_tm1 = states[2]

		z = K.dot(x, self.W) + K.dot(h_tm1, self.U) + K.dot(y_tm1, self.A) + self.b

		z0 = z[:, :self.input_dim]
		z1 = z[:, self.input_dim: 2 * self.input_dim]
		z2 = z[:, 2 * self.input_dim: 3 * self.input_dim]
		z3 = z[:, 3 * self.input_dim:]

		i = self.inner_activation(z0)
		f = self.inner_activation(z1)
		c = f * c_tm1 + i * self.activation(z2)
		o = self.inner_activation(z3)

		h = o * self.activation(c)
		y = self.out_activation(K.dot(h, self.V_y) + self.b_y)

		return y, [h, c, y]

	def get_config(self):
		config = {'output_dim': self.output_dim,
		          'init': self.init.__name__,
		          'inner_init': self.inner_init.__name__,
		          'forget_bias_init': self.forget_bias_init.__name__,
		          'activation': self.activation.__name__,
		          'out_activation': self.out_activation.__name__,
		          'inner_activation': self.inner_activation.__name__}
		base_config = super(DecoderLSTM, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))










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

		x_t = K.dot(h_tm1, self.A) + self.ba
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
