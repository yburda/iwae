import theano
import theano.tensor as T
import numpy as np
import collections
import nnet
import utils
import progressbar
from utils import t_repeat, log_mean_exp, reshape_and_tile_images

log2pi = T.constant(np.log(2*np.pi).astype(theano.config.floatX))

floatX = theano.config.floatX


class UnitGaussianSampler:
    def __init__(self):
        self.params = []

    def samplesIshape_srng(self, shape, srng):
        return srng.normal(shape)

    def log_likelihood_samples(self, samples):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return -log2pi*T.cast(samples.shape[1], floatX)/2 - T.sum(T.sqr(samples), axis=1) / 2


class GaussianSampler:
    def __init__(self, h_network, mean_network, sigma_network):
        self.h_network = h_network
        self.mean_network = mean_network
        self.sigma_network = sigma_network

        self.params = self.h_network.params + self.mean_network.params + self.sigma_network.params

    def mean_sigmaIx(self, x):
        '''Returns the mean and the square root of the covariance of the Gaussian'''
        h = self.h_network.yIx(x)
        mean = self.mean_network.yIx(h)
        sigma = self.sigma_network.yIx(h)

        return mean, sigma

    def samplesImean_sigma_srng(self, mean, sigma, srng):
        unit_gaussian_samples = srng.normal(mean.shape)
        return sigma * unit_gaussian_samples + mean

    def samplesIx_srng(self, x, srng):
        mean, sigma = self.mean_sigmaIx(x)
        return self.samplesImean_sigma_srng(mean, sigma, srng)

    def log_likelihood_samplesImean_sigma(self, samples, mean, sigma):
        return -log2pi*T.cast(samples.shape[1], floatX) / 2 -                \
               T.sum(T.sqr((samples-mean)/sigma) + 2*T.log(sigma), axis=1) / 2

    def log_likelihood_samplesIx(self, samples, x):
        mean, sigma = self.mean_sigmaIx(x)
        return self.log_likelihood_samplesImean_sigma(samples, mean, sigma)

    def first_linear_layer_weights_np(self):
        return self.h_network.first_linear_layer_weights_np()

    @staticmethod
    def random(n_units, mean=None):
        h_network = nnet.random_linear_then_tanh_chain(n_units[:-1])
        mean_network = nnet.Linear.random(n_units[-2], n_units[-1])
        if mean is not None:
            mean_network.b.set_value(mean.astype(floatX))
        sigma_network = nnet.NNet().add_layer(nnet.Linear.random(n_units[-2], n_units[-1])).add_layer(nnet.Exponential())

        return GaussianSampler(h_network, mean_network, sigma_network)


class BernoulliSampler:
    def __init__(self, mean_network):
        self.mean_network = mean_network

        self.params = self.mean_network.params

    def meanIx(self, x):
        return self.mean_network.yIx(x)

    def samplesImean_srng(self, mean, srng):
        return T.cast(T.le(srng.uniform(mean.shape), mean), mean.dtype)

    def samplesIx_srng(self, x, srng):
        return self.samplesImean_srng(self.meanIx(x), srng)

    def log_likelihood_samplesImean(self, samples, mean):
        return T.sum(samples * T.log(mean) + (1 - samples) * T.log(1 - mean), axis=1)

    def log_likelihood_samplesIx(self, samples, x):
        mean = self.meanIx(x)
        return self.log_likelihood_samplesImean(samples, mean)

    def last_linear_layer_weights_np(self):
        return self.mean_network.last_linear_layer_weights_np()

    def first_linear_layer_weights_np(self):
        return self.mean_network.first_linear_layer_weights_np()

    @staticmethod
    def random(n_units, bias=None):
        mean_network = nnet.random_linear_then_tanh_chain(n_units[:-1])

        mean_network.add_layer(nnet.Linear.random(n_units[-2], n_units[-1]))

        if bias is not None:
            mean_network.layers[-1].b.set_value(bias.astype(theano.config.floatX))

        mean_network.add_layer(nnet.Sigmoid())

        return BernoulliSampler(mean_network)


class IWAE:
    def __init__(self, q_layers, p_layers, prior):
        self.q_layers = q_layers
        self.p_layers = p_layers
        self.prior = prior

        self.params = []
        for layer in self.q_layers:
            self.params += layer.params
        for layer in self.p_layers:
            self.params += layer.params
        self.params += prior.params

    def q_samplesIx_srng(self, x, srng):
        samples = [x]
        for layer in self.q_layers:
            samples.append(layer.samplesIx_srng(samples[-1], srng))
        return samples

    def log_weightsIq_samples(self, q_samples):
        log_weights = 0
        for layer_q, layer_p, prev_sample, next_sample in zip(self.q_layers, reversed(self.p_layers), q_samples, q_samples[1:]):
            log_weights += layer_p.log_likelihood_samplesIx(prev_sample, next_sample) -\
                           layer_q.log_likelihood_samplesIx(next_sample, prev_sample)
        log_weights += self.prior.log_likelihood_samples(q_samples[-1])
        return log_weights

    def gradIminibatch_srng(self, x, srng, num_samples, model_type='iwae'):
        # rep_x = T.extra_ops.repeat(x, num_samples, axis=0)
        rep_x = t_repeat(x, num_samples, axis=0)  # works marginally faster than theano's T.extra_ops.repeat
        q_samples = self.q_samplesIx_srng(rep_x, srng)

        log_ws = self.log_weightsIq_samples(q_samples)

        log_ws_matrix = log_ws.reshape((x.shape[0], num_samples))
        log_ws_minus_max = log_ws_matrix - T.max(log_ws_matrix, axis=1, keepdims=True)
        ws = T.exp(log_ws_minus_max)
        ws_normalized = ws / T.sum(ws, axis=1, keepdims=True)
        ws_normalized_vector = T.reshape(ws_normalized, log_ws.shape)

        dummy_vec = T.vector(dtype=theano.config.floatX)

        if model_type in ['vae', 'VAE']:
            print "Training a VAE"
            return collections.OrderedDict([(
                                             param,
                                             T.grad(T.sum(log_ws)/T.cast(num_samples, log_ws.dtype), param)
                                            )
                                            for param in self.params])
        else:
            print "Training an IWAE"
            return collections.OrderedDict([(
                                             param,
                                             theano.clone(
                                                T.grad(T.dot(log_ws, dummy_vec), param),
                                                replace={dummy_vec: ws_normalized_vector})
                                            )
                                            for param in self.params])

    def log_marginal_likelihood_estimate(self, x, num_samples, srng):
        num_xs = x.shape[0]
        # rep_x = T.extra_ops.repeat(x, num_samples, axis=0)
        rep_x = t_repeat(x, num_samples, axis=0)
        samples = self.q_samplesIx_srng(rep_x, srng)

        log_ws = self.log_weightsIq_samples(samples)
        log_ws_matrix = T.reshape(log_ws, (num_xs, num_samples))
        log_marginal_estimate = log_mean_exp(log_ws_matrix, axis=1)

        return log_marginal_estimate

    def first_q_layer_weights_np(self):
        return self.q_layers[0].first_linear_layer_weights_np()

    def last_p_layer_weights_np(self):
        return self.p_layers[-1].last_linear_layer_weights_np()

    def first_p_layer_weights_np(self):
        return self.p_layers[0].first_linear_layer_weights_np()

    @staticmethod
    def random(latent_units, hidden_units_q, hidden_units_p, bias=None, data_type='binary'):
        layers_q = []
        for units_prev, units_next, hidden_units in zip(latent_units, latent_units[1:], hidden_units_q):
            layers_q.append(GaussianSampler.random([units_prev]+hidden_units+[units_next]))

        layers_p = []
        for units_prev, units_next, hidden_units in \
                zip(list(reversed(latent_units))[:-1], list(reversed(latent_units))[1:-1], hidden_units_p[:-1]):
            layers_p.append(GaussianSampler.random([units_prev]+hidden_units+[units_next]))
        if data_type == 'binary':
            layers_p.append(BernoulliSampler.random([latent_units[1]]+hidden_units_p[-1]+[latent_units[0]], bias=bias))
        elif data_type == 'continuous':
            layers_p.append(GaussianSampler.random([latent_units[1]]+hidden_units_p[-1]+[latent_units[0]], bias=bias))

        prior = UnitGaussianSampler()
        return IWAE(layers_q, layers_p, prior)


def random_iwae(latent_units, hidden_units_q, hidden_units_p, dataset):
    return IWAE.random(latent_units, hidden_units_q, hidden_units_p,
                       bias=dataset.get_train_bias_np())


def get_samples(model, num_samples, seed=123):
    srng = utils.srng(seed)
    prior_samples = model.prior.samplesIshape_srng((num_samples, model.first_p_layer_weights_np().shape[0]), srng)
    samples = [prior_samples]
    for layer in model.p_layers[:-1]:
        samples.append(layer.samplesIx_srng(samples[-1], srng))
    samples_function = theano.function([], model.p_layers[-1].meanIx(samples[-1]))

    return reshape_and_tile_images(samples_function())


def measure_marginal_log_likelihood(model, dataset, subdataset, seed=123, minibatch_size=20, num_samples=50):
    print "Measuring {} log likelihood".format(subdataset)
    srng = utils.srng(seed)
    test_x = dataset.data[subdataset]
    n_examples = test_x.get_value(borrow=True).shape[0]

    if n_examples % minibatch_size == 0:
        num_minibatches = n_examples // minibatch_size
    else:
        num_minibatches = n_examples // minibatch_size + 1

    index = T.lscalar('i')
    minibatch = dataset.minibatchIindex_minibatch_size(index, minibatch_size, subdataset=subdataset, srng=srng)

    log_marginal_likelihood_estimate = model.log_marginal_likelihood_estimate(minibatch, num_samples, srng)

    get_log_marginal_likelihood = theano.function([index], T.sum(log_marginal_likelihood_estimate))

    pbar = progressbar.ProgressBar(maxval=num_minibatches).start()
    sum_of_log_likelihoods = 0.
    for i in xrange(num_minibatches):
        summand = get_log_marginal_likelihood(i)
        sum_of_log_likelihoods += summand
        pbar.update(i)
    pbar.finish()

    marginal_log_likelihood = sum_of_log_likelihoods/n_examples

    return marginal_log_likelihood


def get_first_q_layer_weights(model):
    return utils.reshape_and_tile_images(model.first_q_layer_weights_np().T)

def get_last_p_layer_weights(model):
    return utils.reshape_and_tile_images(model.last_p_layer_weights_np())


def get_units_variances(model, dataset):
    srng = utils.srng()

    x = dataset.minibatchIindex_minibatch_size(0, 500, subdataset='train', srng=srng)

    samples = model.q_samplesIx_srng(x, srng)

    means = []
    for layer, x in zip(model.q_layers, samples):
        mean, _ = layer.mean_sigmaIx(x)
        means.append(mean)

    mean_fun = theano.function([], means)
    mean_vals = mean_fun()

    vars_of_means = [np.var(mean_val, axis=0) for mean_val in mean_vals]

    return vars_of_means


def chop_units_with_variance_under_threshold(model, variances, threshold=0.01):
    ords = [np.argsort(stds_mean)[::-1] for stds_mean in variances]

    indices = []
    for order, var in zip(ords, variances):
        ordered_var = var[order[::-1]]
        last_index = np.searchsorted(ordered_var, threshold)
        indices.append(order[:order.shape[0]-last_index])

    num_units_in_input = model.q_layers[0].first_linear_layer_weights_np().shape[0]
    for q_layer, p_layer, ord_incoming, ord_outcoming in\
            zip(model.q_layers, reversed(model.p_layers), [np.arange(num_units_in_input)]+indices, indices):
        mean_net = q_layer.mean_network
        sigma_net = q_layer.sigma_network
        h_net = q_layer.h_network
        mean_net.W.set_value(mean_net.W.get_value()[:, ord_outcoming])
        mean_net.b.set_value(mean_net.b.get_value()[:, ord_outcoming])
        sigma_net.layers[0].W.set_value(sigma_net.layers[0].W.get_value()[:, ord_outcoming])
        sigma_net.layers[0].b.set_value(sigma_net.layers[0].b.get_value()[:, ord_outcoming])
        h_net.layers[0].W.set_value(h_net.layers[0].W.get_value()[ord_incoming, :])

        if isinstance(p_layer, BernoulliSampler):
            mean_net = p_layer.mean_network

            mean_net.layers[0].W.set_value(mean_net.layers[0].W.get_value()[ord_outcoming, :])
        elif isinstance(p_layer, GaussianSampler):
            mean_net = p_layer.mean_network
            sigma_net = p_layer.sigma_network
            h_net = p_layer.h_network
            mean_net.W.set_value(mean_net.W.get_value()[:, ord_incoming])
            mean_net.b.set_value(mean_net.b.get_value()[:, ord_incoming])
            sigma_net.layers[0].W.set_value(sigma_net.layers[0].W.get_value()[:, ord_incoming])
            sigma_net.layers[0].b.set_value(sigma_net.layers[0].b.get_value()[:, ord_incoming])
            h_net.layers[0].W.set_value(h_net.layers[0].W.get_value()[ord_outcoming, :])
