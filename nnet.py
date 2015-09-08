'''Classes used for describing and creating neural networks'''

import theano
import theano.tensor as T
import numpy as np
floatX = theano.config.floatX


class Linear():
    def __init__(self, W, b):
        self.W = theano.shared(value=W.astype(floatX), name='W')
        self.b = theano.shared(value=b.astype(floatX), name='b', broadcastable=(True, False))
        self.params = [self.W, self.b]

    def yIx(self, x, **kwargs):
        return x.dot(self.W) + self.b

    def first_linear_layer_weights_np(self):
        return self.W.get_value(borrow=False)
    def last_linear_layer_weights_np(self):
        return self.W.get_value(borrow=False)

    @staticmethod
    def random(n_in, n_out, factor=1., seed=123):
        '''A randomly initialized linear layer.
        When factor is 1, the initialization is uniform as in Glorot, Bengio, 2010,
        assuming the layer is intended to be followed by the tanh nonlinearity.'''
        random_state = np.random.RandomState(seed)

        scale = factor*np.sqrt(6./(n_in+n_out))
        return Linear(W=random_state.uniform(low=-scale,
                                             high=scale,
                                             size=(n_in, n_out)),
                      b=np.zeros((1, n_out)))


class Tanh():
    def __init__(self):
        self.params = []

    def yIx(self, x, **kwargs):
        return T.tanh(x)


class Sigmoid():
    def __init__(self):
        self.params = []

    def yIx(self, x, **kwargs):
        return T.nnet.sigmoid(x)


class Exponential():
    def __init__(self):
        self.params = []

    def yIx(self, x, **kwargs):
        return T.exp(x)


class NNet():
    def __init__(self):
        self.params = []
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)
        self.params += layer.params
        return self

    def yIx(self, x, **kwargs):
        '''Returns the output of the last layer of the network'''
        y = 1 * x
        for layer in self.layers:
            y = layer.yIx(y, **kwargs)
        return y

    def first_linear_layer_weights_np(self):
        first_linear_layer = next(layer for layer in self.layers if isinstance(layer, Linear))
        return first_linear_layer.W.get_value(borrow=False)

    def last_linear_layer_weights_np(self):
        last_linear_layer = [layer for layer in self.layers if isinstance(layer, Linear)][-1]
        return last_linear_layer.W.get_value(borrow=False)


def random_linear_then_tanh_chain(n_units):
    '''Returns a neural network consisting of alternating Linear and Tanh layers.'''
    model = NNet()
    for n_in, n_out in zip(n_units, n_units[1:]):
        model.add_layer(Linear.random(n_in, n_out))
        model.add_layer(Tanh())
    return model
