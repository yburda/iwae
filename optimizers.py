from utils import shared_zeros_like, shared_ones_like

import numpy as np
import theano
import theano.tensor as T

import collections


class Adam():
    def __init__(self, model, learning_rate=0.001, beta_1=0.99, beta_2=0.999, timestep=1):
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_moments = collections.OrderedDict([(param, shared_zeros_like(param)) for param in model.params])
        self.second_moments = collections.OrderedDict([(param, shared_ones_like(param)) for param in model.params])
        self.timestep = theano.shared(np.array(timestep).astype(theano.config.floatX))

    def updatesIgrad_model(self, grad, model):
        timestep_update = collections.OrderedDict([(self.timestep, self.timestep+1)])

        first_moments_updates = collections.OrderedDict([(self.first_moments[param],
                                                          self.beta_1 * self.first_moments[param] + (1-self.beta_1) * grad[param])
                                                         for param in model.params])

        second_moments_updates = collections.OrderedDict([(self.second_moments[param],
                                                          self.beta_2 * self.second_moments[param] + (1-self.beta_2) * T.sqr(grad[param]))
                                                         for param in model.params])

        alpha = self.learning_rate * T.sqrt(1 - self.beta_2 ** self.timestep) / (1 - self.beta_1 ** self.timestep)

        sgd_updates = collections.OrderedDict([(param,
                                                param + alpha * self.first_moments[param] / (T.sqrt(self.second_moments[param]) + 1e-4))
                                               for param in model.params])

        new_ordered_dict = collections.OrderedDict()
        new_ordered_dict.update(timestep_update)
        new_ordered_dict.update(first_moments_updates)
        new_ordered_dict.update(second_moments_updates)
        new_ordered_dict.update(sgd_updates)
        return new_ordered_dict
