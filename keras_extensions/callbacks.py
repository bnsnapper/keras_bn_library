import sys
import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from keras.models import Model, Sequential

class UnsupervisedLoss2Logger(Callback):
	def __init__(self, X_train, X_test, loss, verbose=1, batch_size = 1,
	         label='loss', every_n_epochs=1, display_delta=True):
		super(UnsupervisedLoss2Logger, self).__init__()
		self.X_train = X_train
		self.X_test = X_test
		self.loss = loss
		self.verbose = verbose
		self.label = label
		self.every_n_epochs = every_n_epochs
		self.display_delta = display_delta
		self.prev_loss = None
		self.batch_size = batch_size

		input_train = K.placeholder(shape=self.X_train.shape)
		input_test = K.placeholder(shape=self.X_test.shape)
		loss = self.loss(input_train, input_test)
		ins = [input_train, input_test]
		self.loss_function = K.function(ins, loss)

	def on_epoch_end(self, epoch, logs={}):
		if((epoch+1) % self.every_n_epochs == 0):

			loss = np.mean(self.loss_function([self.X_train, self.X_test]))

			if(self.prev_loss):
				delta_loss = loss - self.prev_loss
			else:
				delta_loss = None
			self.prev_loss = loss

			if(self.display_delta and delta_loss):
				print(' - %s: %f (%+f)' % (self.label, loss, delta_loss))
			else:
				print(' - %s: %f' % (self.label, loss))
			#sys.stdout.flush()

