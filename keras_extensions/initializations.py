from __future__ import absolute_import
import numpy as np
from keras.initializations import get_fans, uniform

def glorot_uniform_sigm(shape, name=None, dim_ordering='th'):
    """
    Glorot style weight initializer for sigmoid activations.
    
    Like keras.initializations.glorot_uniform(), but with uniform random interval like in 
    Deeplearning.net tutorials.
    They claim that the initialization random interval should be
      +/- sqrt(6 / (fan_in + fan_out)) (like Keras' glorot_uniform()) when tanh activations are used, 
      +/- 4 sqrt(6 / (fan_in + fan_out)) when sigmoid activations are used.
    See: http://deeplearning.net/tutorial/mlp.html#going-from-logistic-regression-to-mlp
    """
    fan_in, fan_out = get_fans(shape, dim_ordering=dim_ordering)
    s = 4. * np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s, name=name)
