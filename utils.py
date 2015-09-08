import math
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams


def srng(seed=123):
    return MRG_RandomStreams(seed=seed)


def t_repeat(x, num_repeats, axis):
    '''Repeats x along an axis num_repeats times. Axis has to be 0 or 1, x has to be a matrix.'''
    if num_repeats == 1:
        return x
    else:
        if axis == 0:
            return T.alloc(x.dimshuffle(1, 0, 'x'), x.shape[1], x.shape[0], num_repeats)\
                   .reshape((x.shape[1], num_repeats*x.shape[0]))\
                   .dimshuffle(1, 0)
        elif axis == 1:
            return T.alloc(x.dimshuffle(0, 'x', 1), x.shape[0], num_repeats, x.shape[1]).reshape((x.shape[0], num_repeats*x.shape[1]))


def t_mean(x, axis=None, keepdims=False):
    if axis is None:
        return T.sum(x, keepdims=keepdims) / T.cast(x.shape[0]*x.shape[1], x.dtype)
    else:
        return T.sum(x, axis=axis, keepdims=keepdims) / T.cast(x.shape[axis], x.dtype)


def log_mean_exp(x, axis):
    m = T.max(x, axis=axis, keepdims=True)
    return m + T.log(T.mean(T.exp(x - m), axis=axis, keepdims=True))


def shared_zeros_like(shared_var):
    return theano.shared(np.zeros(shared_var.get_value(borrow=True).shape).astype(shared_var.dtype),
                         broadcastable=shared_var.broadcastable)


def shared_ones_like(shared_var):
    return theano.shared(np.ones(shared_var.get_value(borrow=True).shape).astype(shared_var.dtype),
                         broadcastable=shared_var.broadcastable)


def reshape_and_tile_images(array, shape=(28, 28), n_cols=None):
    if n_cols is None:
        n_cols = int(math.sqrt(array.shape[0]))
    n_rows = int(math.ceil(float(array.shape[0])/n_cols))

    def cell(i, j):
        ind = i*n_cols+j
        if i*n_cols+j < array.shape[0]:
            return array[ind].reshape(*shape, order='C')
        else:
            return np.zeros(shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)
