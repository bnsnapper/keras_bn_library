from __future__ import absolute_import

from keras import backend as K

def nrlu(x):
   std = K.mean(K.sigmoid(x))
   eta = K.random_normal(shape=x.shape, std=std)
   y = K.maximum(x + eta, 0)
   return y

