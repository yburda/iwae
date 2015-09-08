import theano
import theano.tensor as T

import progressbar


def train(model, dataset, optimizer, minibatch_size, n_epochs, srng, **kwargs):
    print "training for {} epochs with {} learning rate".format(n_epochs, optimizer.learning_rate)
    num_minibatches = dataset.get_n_examples('train') / minibatch_size

    index = T.lscalar('i')
    minibatch = dataset.minibatchIindex_minibatch_size(index, minibatch_size, srng=srng, subdataset='train')

    grad = model.gradIminibatch_srng(minibatch, srng, **kwargs)
    updates = optimizer.updatesIgrad_model(grad, model)

    train_step = theano.function([index], None, updates=updates)

    pbar = progressbar.ProgressBar(maxval=n_epochs*num_minibatches).start()
    for j in xrange(n_epochs):
        for i in xrange(num_minibatches):
            train_step(i)
            pbar.update(j*num_minibatches+i)
    pbar.finish()
    return model
