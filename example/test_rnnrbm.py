
from __future__ import print_function
import numpy as np
np.random.seed(1234)

from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.layers import Dense, Flatten
from keras_extensions.rbm import RBM
from keras_extensions.initializations import glorot_uniform_sigm
from keras_extensions.rnnrbm import RNNRBM
import matplotlib.pyplot as plt

# since we are using stateful rnn tsteps can be set to 1
tsteps = 1
lahead = 1
batch_size = 25
epochs = 10
epochs_ft = 25

hidden_dim = 10
hidden_recurrent_dim = 50

lr = 1e-4

stateful= True

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
cos2 = cos.reshape(cos.shape[0], cos.shape[2])
input_dim = cos.shape[2]
print('Input shape:', cos.shape)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos) - lahead):
	expected_output[i, 0] = np.mean(cos[i + 1:i + lahead + 1])

print('Output shape')
print(expected_output.shape)

print('Training Pretrain RBM')
rbm = RBM(hidden_dim,
		init=glorot_uniform_sigm,
		input_dim=input_dim,
		hidden_unit_type='binary',
		visible_unit_type='gaussian',
		persistent=True,
		batch_size=batch_size,
		nb_gibbs_steps=10
		)

model = Sequential()
model.add(Flatten(batch_input_shape=(batch_size, tsteps, input_dim)))
model.add(rbm)

opt = RMSprop(lr=lr)
model.compile(loss=rbm.contrastive_divergence_loss,
			  optimizer=opt,
			  metrics=[rbm.reconstruction_loss])
model.summary()

for i in range(epochs):
	print('Epoch', i, '/', epochs)
	model.fit(cos,
              cos2,
	          batch_size=batch_size,
	          verbose=1,
	          nb_epoch=1,
	          shuffle=False)
	if(stateful):
		model.reset_states()



print('Training Pretrain RNNRBM')
rnnrbm = RNNRBM(hidden_dim, hidden_recurrent_dim,
				batch_input_shape=(batch_size, tsteps, input_dim),
				return_sequences=False,
				stateful=stateful,
				activation="tanh",
				nb_gibbs_steps=10,
				persistent=True,
				rbm=rbm,
				dropout_U=0.0, dropout_W=0.0, dropout_RBM=0.0)

model = Sequential()
model.add(rnnrbm)

model.compile(loss=rnnrbm.loss,
			  optimizer=opt,
			  metrics=[rnnrbm.metrics])
model.summary()

for i in range(epochs):
	print('Epoch', i, '/', epochs)
	model.fit(cos,
              cos2,
	          batch_size=batch_size,
	          verbose=1,
	          nb_epoch=1,
	          shuffle=False)
	if(stateful):
		model.reset_states()


print("Fine Tuning")
rnnrbm.set_finetune()

model = Sequential()
model.add(rnnrbm)
model.add(Dense(1))
model.compile(loss="mse", optimizer=opt)
model.summary()

print('Training Finetune')
for i in range(epochs_ft):
	print('Epoch', i, '/', epochs_ft)
	model.fit(cos,
				expected_output,
				batch_size=batch_size,
				verbose=1,
				nb_epoch=1,
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
