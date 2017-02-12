from __future__ import print_function
import numpy as np
np.random.seed(1234)

import keras.backend as K
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers.core import Dense
from keras_extensions.recurrent import QRNN
import matplotlib.pyplot as plt

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
lahead = 1
batch_size = 25
nb_epoch = 50

stateful=False

lstm_dim = 50
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
	cos = np.zeros(((xn - x0) * step, 1, 1), dtype=K.floatx())
	for i in range(len(cos)):
		idx = x0 + i * step
		cos[i, 0, 0] = amp * np.cos(2 * np.pi * idx / period)
		cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-k * idx)
	return cos


print('Generating Data')
cos = gen_cosine_amp()
input_dim = cos.shape[2]
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1), dtype=K.floatx())
for i in range(len(cos) - lahead):
	expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Creating Model')
model = Sequential()
model.add(QRNN(lstm_dim,
				activation='linear',
				return_sequences=False, stateful=stateful,
				batch_input_shape=(batch_size, tsteps, input_dim)))
model.add(Dense(1))
opt = RMSprop(lr=lr)

print('Training Model')
model.compile(optimizer=opt, loss="mse")
model.summary()
for i in range(nb_epoch):
	print('Epoch', i, '/', nb_epoch)
	model.fit(cos, expected_output,
    		    batch_size=batch_size,
	    	    nb_epoch=1,
				verbose=1,	
	    	    shuffle=False)
	if(stateful):
		model.reset_states()


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
