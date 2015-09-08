import datasets
import iwae
import optimizers
import train
import utils
import config

import os
import cPickle as pkl
import argparse
import numpy as np


def save_checkpoint(directory_name, i, model, optimizer, srng):
    '''Saves model, optimizer, and random number generator state srng.state as a pickle file named training_state[i].pkl'''
    try:
        filename_to_save = os.path.join(directory_name, "training_state{}.pkl".format(i))
        with open(filename_to_save, "wb") as f:
            pkl.dump([model, optimizer, srng.rstate], f, protocol=pkl.HIGHEST_PROTOCOL)
    except:
        print "Failed to write to file {}".format(filename_to_save)


def load_checkpoint(directory_name, i):
    '''Loads model, optimizer, and random number generator from a pickle file named training_state[i].pkl
    Returns -1, None, None, None if loading failed
    Returns i, model, optimizer, random number generator if loading succeedeed'''
    try:
        load_from_filename = os.path.join(directory_name, "training_state{}.pkl".format(i))

        with open(load_from_filename, "rb") as f:
            model, optimizer, rstate = pkl.load(f)
        srng = utils.srng()
        srng.rstate = rstate
        loaded_checkpoint = i
    except:
        loaded_checkpoint = -1
        model, optimizer, srng = None, None, None
    return loaded_checkpoint, model, optimizer, srng


def post_experiment(directory_name, dataset, model):
    '''Analyze the model: draw samples, measure test and training log likelihoods'''
    samples = iwae.get_samples(model, 100)
    q_weights = iwae.get_first_q_layer_weights(model)
    p_weights = iwae.get_last_p_layer_weights(model)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imshow(samples, cmap='Greys')
    plt.savefig(os.path.join(directory_name, "samples.jpg"))
    plt.close()
    plt.imshow(q_weights, cmap='Greys')
    plt.savefig(os.path.join(directory_name, "q_weights.jpg"))
    plt.close()
    plt.imshow(p_weights, cmap='Greys')
    plt.savefig(os.path.join(directory_name, "p_weights.jpg"))
    plt.close()

    num_samples = 5000
    marginal_log_likelihood = iwae.measure_marginal_log_likelihood(model=model, dataset=dataset,
                                                                   subdataset="test", num_samples=num_samples)

    with open(os.path.join(directory_name, "test_log_likelihood_{}_samples.txt".format(num_samples)), "w") as f:
        f.write(str(marginal_log_likelihood))
    print marginal_log_likelihood

    marginal_log_likelihood = iwae.measure_marginal_log_likelihood(model=model, dataset=dataset,
                                                                   subdataset="train", num_samples=num_samples)

    with open(os.path.join(directory_name, "train_log_likelihood_{}_samples.txt".format(num_samples)), "w") as f:
        f.write(str(marginal_log_likelihood))
    print marginal_log_likelihood

    variances = iwae.get_units_variances(model, dataset)
    for i, var in enumerate(variances):
        plt.hist(np.log(var), bins=20)
        plt.savefig(os.path.join(directory_name, "log_variances_layer_{}.png".format(i+1)))
        plt.close()
    iwae.chop_units_with_variance_under_threshold(model, variances)

    with open(os.path.join(directory_name, "numbers_of_active_units.txt".format(num_samples)), "w") as f:
        f.write(str([layer.mean_network.last_linear_layer_weights_np().shape[1] for layer in model.q_layers]))

    marginal_log_likelihood = iwae.measure_marginal_log_likelihood(model=model, dataset=dataset,
                                                                   subdataset="test", num_samples=num_samples)

    with open(os.path.join(directory_name, "test_log_likelihood_{}_samples_dead_units_removed.txt".format(num_samples)), "w") as f:
        f.write(str(marginal_log_likelihood))
    print marginal_log_likelihood


def directory_to_store(**kwargs):
    '''Expects arguments that describe the experiment and returns the directory where the results of the experiment should be stored'''
    if kwargs['exp'] == 'train':
        directory_name = '{}l{}{}k{}'.format(kwargs['dataset'], kwargs['layers'], kwargs['model'], kwargs['k'])
    else:
        directory_name = kwargs['exp']

    return os.path.join(config.RESULTS_DIR, directory_name)


def training_experiment(directory_name, latent_units, hidden_units_q, hidden_units_p, k, model_type, dataset, checkpoint=-1):
    '''The experiment that trains a model with given parameters'''
    def checkpoint0(dataset):
        data_dimension = dataset.get_data_dim()
        model = iwae.random_iwae(latent_units=[data_dimension] + latent_units,
                                 hidden_units_q=hidden_units_q,
                                 hidden_units_p=hidden_units_p,
                                 dataset=dataset
                                 )
        srng = utils.srng()
        optimizer = optimizers.Adam(model=model, learning_rate=1e-3)

        return model, optimizer, srng

    def checkpoint1to8(i, dataset, model, optimizer, srng):
        optimizer.learning_rate = 1e-4*round(10.**(1-(i-1)/7.), 1)
        model = train.train(model=model, dataset=dataset, optimizer=optimizer,
                            minibatch_size=20, n_epochs=3**(i-1), srng=srng,
                            num_samples=k, model_type=model_type)
        return model, optimizer, srng

    dataset = datasets.load_dataset_from_name(dataset)

    loaded_checkpoint = -1
    if checkpoint >= 0:
        loaded_checkpoint, model, optimizer, srng = load_checkpoint(directory_name, checkpoint)
        if loaded_checkpoint == -1:
            print "Unable to load checkpoint {} from {}, starting the experiment from the beginning".format(checkpoint, directory_name)

    if loaded_checkpoint < 0:
        model, optimizer, srng = checkpoint0(dataset)
        save_checkpoint(directory_name, 0, model, optimizer, srng)
        loaded_checkpoint = 0

    for i in range(loaded_checkpoint+1, 9):
        model, optimizer, srng = checkpoint1to8(i, dataset, model, optimizer, srng)
        save_checkpoint(directory_name, i, model, optimizer, srng)
    loaded_checkpoint = 8

    post_experiment(directory_name, dataset, model)


def experiment2(directory_name, dataset='MNIST', direction='vae_to_iwae'):
    '''The experiment that trains a vae initialized at an iwae or vice versa'''
    dataset = datasets.load_dataset_from_name(dataset)
    if direction == 'vae_to_iwae':
        previous_args = dict(layers=1, model='vae', k=1, dataset='MNIST', exp='train')
        new_model_type = 'iwae'
        new_k = 50
    elif direction == 'iwae_to_vae':
        previous_args = dict(layers=1, model='iwae', k=50, dataset='MNIST', exp='train')
        new_model_type = 'vae'
        new_k = 1
    previous_directory_name = directory_to_store(**previous_args)
    loaded_checkpoint, model, optimizer, srng = load_checkpoint(previous_directory_name, 8)
    optimizer.learning_rate = 1e-4
    model = train.train(model=model, dataset=dataset, optimizer=optimizer,
                        minibatch_size=20, n_epochs=3**7, srng=srng,
                        num_samples=new_k, model_type=new_model_type)
    save_checkpoint(directory_name, 0, model, optimizer, srng)

    post_experiment(directory_name, dataset, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run VAE/IWAE training experiments.')
    parser.add_argument('--exp', '-e', choices=['train', 'vae_to_iwae', 'iwae_to_vae'], default='train')
    parser.add_argument('--layers', '-l', type=int, choices=[1, 2], default=1)
    parser.add_argument('--model', '-m', choices=['vae', 'iwae'], default='vae')
    parser.add_argument('--k', '-k', type=int, default=1)
    parser.add_argument('--dataset', '-d', choices=['MNIST', 'OMNI', 'BinFixMNIST'], default='MNIST')
    parser.add_argument('--checkpoint', '-c', type=int, default=-1)

    args = parser.parse_args()
    if args.layers == 1:
        latent_units = [50]
        hidden_units_q = [[200, 200]]
        hidden_units_p = [[200, 200]]
    elif args.layers == 2:
        latent_units = [100, 50]
        hidden_units_q = [[200, 200], [100, 100]]
        hidden_units_p = [[100, 100], [200, 200]]
    directory_name = directory_to_store(**args.__dict__)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    if args.exp == 'train':
        training_experiment(directory_name, latent_units=latent_units, hidden_units_q=hidden_units_q, hidden_units_p=hidden_units_p,
                            k=args.k, model_type=args.model, dataset=args.dataset, checkpoint=args.checkpoint)
    else:
        experiment2(directory_name, direction=args.exp, dataset=args.dataset)
