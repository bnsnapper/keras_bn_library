
from __future__ import print_function
import numpy as np
np.random.seed(1234)

import keras.backend as K
from keras import objectives
from keras.models import Sequential, Model
from keras.layers import Input, RepeatVector
from keras.optimizers import SGD, RMSprop
from keras.layers.core import Dense, Lambda, Flatten
from keras.layers.recurrent import LSTM
from keras_extensions.recurrent import DecoderVaeLSTM
import matplotlib.pyplot as plt

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
lahead = 1
batch_size = 25
nb_epoch_ae = 50
nb_epoch_ft = 25

latent_dim = 10
lstm_dim = 100
epsilon_std = 1.0
lr = 1e-4

def gen_cosine_amp(amp=100, period=1000, x0=0, xn=50000, step=1, k=0.0001):
	"""Generates an absolute cosine time series with the amplitude
	exponentially decreasing

	Arguments:
	    amp: amplitude of the cosine function
	    period: period of the cosine function
	    x0: initial x of the time series
	    xn: final x of the time series
	    step: step of the time series discretization
	    k: exponential rate
	"""
	cos = np.zeros(((xn - x0) * step, 1, 1))
	for i in range(len(cos)):
		idx = x0 + i * step
		cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
		cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
	return cos


print('Generating Data')
cos = gen_cosine_amp()
input_dim = cos.shape[2]
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
	expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
x = Input(batch_shape=(batch_size, tsteps, input_dim))
h = LSTM(lstm_dim, stateful=False, activation='linear')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)
output=Dense(input_dim)(z_mean)

def sampling(args):
	z_mean, z_log_var = args
	epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
			                         std=epsilon_std)
	return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoder_z = Dense(lstm_dim)(z)
rv = RepeatVector(tsteps)(decoder_z)
decoder_out = DecoderVaeLSTM(input_dim, stateful=False, return_sequences=True, activation='linear')(rv)

def vae_loss(x, x_decoded_mean):
	x_d = Flatten()(x)
	x_dec_d = Flatten()(x_decoded_mean)
	xent_loss = input_dim * objectives.mean_squared_error(x_d, x_dec_d) 
	kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) 
	return 	xent_loss + kl_loss 

vae = Model(x, decoder_out) 
opt = RMSprop(lr=lr)

model = Model(x, output)

print('Training Variational Auto-Encoder LSTM')
vae.compile(optimizer=opt, loss=vae_loss) 
vae.summary()

for i in range(nb_epoch_ae):
	print('Epoch', i, '/', nb_epoch_ae)
	vae.fit(cos, cos,
		    batch_size=batch_size, 
		    nb_epoch=1,
	        verbose=1,
		    shuffle=False)

print('Training Inference Model')
model.compile(optimizer=opt, loss="mse")
model.summary()
for i in range(nb_epoch_ft):
	print('Epoch', i, '/', nb_epoch_ft)
	model.fit(cos, expected_output,
    		    batch_size=batch_size, 
	    	    nb_epoch=1,
				verbose=1,	
	    	    shuffle=False)


print('Predicting')
predicted_output = model.predict(cos, batch_size=batch_size)

print('Plotting Results')
plt.subplot(2, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(2, 1, 2)
plt.plot(predicted_output)
plt.title('Predicted')
plt.show()
