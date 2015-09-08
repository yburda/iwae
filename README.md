# IWAE
This is code accompanying the paper Importance Weighted Autoencoders by Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov available at http://arxiv.org/abs/1509.00519.

# Installation and Requirements
This implementation is based on [theano](http://deeplearning.net/software/theano/). It also uses [progressbar](https://pypi.python.org/pypi/progressbar) to display the training progress.

# Configuration
The file config.py contains variables DATASETS_DIR and RESULTS_DIR. The results of all the experiments will be stored in the folder RESULTS_DIR. The scripts will be looking for the datasets in the directory DATASETS_DIR.

# Datasets
To download the MNIST dataset (both the non-binarized version, and the binarized version of [Larochelle, Murray](http://jmlr.csail.mit.edu/proceedings/papers/v15/larochelle11a/larochelle11a.pdf))
    python download_mnist.py

The OMNIGLOT dataset resized to 28x28 pixel images is provided in the datasets/OMNI/chardata.mat file.

# Usage
## Reproducing results from table 1
To train a model as reported in table 1 of the IWAE paper, run
    python experiments.py --model _model_ --dataset _dataset_ --k _k_ --layers _l_
where __model__ is vae or iwae; _dataset_ is one of BinFixMNIST, MNIST, OMNI; _k_ is 1, 5, or 50; _l_ is 1 or 2.

## Reproducing results from section 5.2
To train a one layer IWAE with k=50 initialized at a one layer VAE trained with k=1, train the one layer VAE with k=1 as described above, and then run
    python experiments.py --exp iwae_to_vae

To train a one layer VAE with k=1 initialized at the one layer IWAE trained with k=50, train the one layer IWAE with k=50 as described above, and then run
    python experiments.py --exp vae_to_iwae

## Checkpoints
To restart a training experiment from a checkpoint created after training for 3<sup>i</sup> epochs, pass the parameter --checkpoint i+1 to the experiments.py script.