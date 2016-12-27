from __future__ import division

import numpy as np
np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import RBM
from keras_extensions.dbn import DBN
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm

# configuration
input_dim = 100
hidden_dim = [500, 200, 10]

dropouts = [0.0, 0.0, 0.0]

batch_size = 10
nb_epoch = [30, 15, 150]
nb_gibbs_steps = 10
lr = 0.001  # small learning rate for GB-RBM

@log_to_file('example_dbn.log')
def main():
	# generate dummy dataset
	nframes = 10000
	dataset = np.random.normal(loc=np.zeros(input_dim), scale=np.ones(input_dim), size=(nframes, input_dim))

	# split into train and test portion
	ntest   = 1000
	X_train = dataset[:-ntest :]     # all but last 1000 samples for training
	X_test  = dataset[-ntest:, :]    # last 1000 samples for testing
	assert X_train.shape[0] >= X_test.shape[0], 'Train set should be at least size of test set!'

	# setup model structure
	print('Creating training model...')
	rbm1=RBM(hidden_dim[0], input_dim=input_dim, init=glorot_uniform_sigm,
				visible_unit_type='gaussian',
				hidden_unit_type='binary',
				nb_gibbs_steps=nb_gibbs_steps, 
				persistent=True, batch_size=batch_size,
				dropout=dropouts[0])

	rbm2=RBM(hidden_dim[1], input_dim=hidden_dim[0], init=glorot_uniform_sigm,
				visible_unit_type='binary',
				hidden_unit_type='binary',
				nb_gibbs_steps=nb_gibbs_steps, 
				persistent=True, batch_size=batch_size,
				dropout=dropouts[1])

	#When using nrlu unit, nb_gibbs_steps and persistent param are ignored
	rbm3=RBM(hidden_dim[2], input_dim=hidden_dim[1], init=glorot_uniform_sigm,
				visible_unit_type='binary',
				hidden_unit_type='nrlu',
				nb_gibbs_steps=1, 
				persistent=False, batch_size=batch_size,
				dropout=dropouts[2])

	rbms=[rbm1, rbm2, rbm3]
	dbn = DBN(rbms)

	# setup optimizer, loss
	def get_layer_loss(rbm,layer_no):
		return rbm.contrastive_divergence_loss
	def get_layer_optimizer(layer_no):
		return SGD((layer_no+1)*lr, 0., decay=0.0, nesterov=False)
	metrics=[]
	for rbm in rbms:
		metrics.append([rbm.reconstruction_loss])
	dbn.compile(layer_optimizer=get_layer_optimizer, layer_loss=get_layer_loss,
		metrics=metrics)

	# do training
	print('Training...')

	dbn.fit(X_train, batch_size, nb_epoch, verbose=1, shuffle=False)

	# generate hidden features from input data
	print('Creating inference model...')
	F= dbn.get_forward_inference_layers()
	B= dbn.get_backward_inference_layers()
	inference_model = Sequential()
	for f in F:
		inference_model.add(f)
		inference_model.add(SampleBernoulli(mode='random'))
	for b in B[:-1]:
		inference_model.add(b)
		inference_model.add(SampleBernoulli(mode='random'))
	# last layer is a gaussian layer
	inference_model.add(B[-1])

	print('Compiling Theano graph...')
	opt = SGD()
	inference_model.compile(opt, loss='mean_squared_error')

	print('Doing inference...')
	h = inference_model.predict(dataset)

	print(h)

	print('Done!')

if __name__ == '__main__':
	main()
