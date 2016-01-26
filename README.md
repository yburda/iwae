# IWAE
This is the code accompanying the paper [Importance Weighted Autoencoders](http://arxiv.org/abs/1509.00519) by Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov.

# Requirements
This implementation is based on [theano](http://deeplearning.net/software/theano/). It also uses [progressbar](https://pypi.python.org/pypi/progressbar) to display the training progress.

# Configuration
The file config.py contains variables DATASETS_DIR and RESULTS_DIR. The results of all the experiments will be stored in the folder RESULTS_DIR. The scripts will be looking for the datasets in the directory DATASETS_DIR.

# Datasets
To download the MNIST dataset (both the non-binarized version, and the binarized version of [Larochelle, Murray](http://jmlr.csail.mit.edu/proceedings/papers/v15/larochelle11a/larochelle11a.pdf))
```
python download_mnist.py
```

The OMNIGLOT dataset can be found at https://github.com/brendenlake/omniglot. We used a version of the dataset resized to 28x28 pixels available in this repository.

# Usage
## Training randomly initialized models
To train a model (VAE or IWAE) as reported in table 1 in the IWAE paper, run
```
python experiments.py --model [model] --dataset [dataset] --k [k] --layers [l]
```
where _model_ is vae or iwae; _dataset_ is one of BinFixMNIST, MNIST, OMNI; _k_ is 1, 5, or 50; _l_ is 1 or 2.

## Training a model initialized at another trained model
To train a one layer IWAE with k=50 initialized at a one layer VAE trained with k=1, train the one layer VAE with k=1 as described above, and then run
```
python experiments.py --exp iwae_to_vae
```

To train a one layer VAE with k=1 initialized at the one layer IWAE trained with k=50, train the one layer IWAE with k=50 as described above, and then run
```
python experiments.py --exp vae_to_iwae
```

## Checkpoints
To restart a training experiment from a checkpoint created after training for 3<sup>i</sup> epochs, pass the parameter --checkpoint i+1 to the experiments.py script.

# Remarks
The results of the experiments in the first version of the paper are not exactly the same as the results of running the provided code. This is because the experiments in the first version of the paper used code that handled random numbers in a different way than the published code. The second version of the paper will be reporting numbers produced by running the published version of the code, making the results fully reproducible.

The first version of the paper did not report results on the fixed binarization of the MNIST dataset used by [Larochelle, Murray](http://jmlr.csail.mit.edu/proceedings/papers/v15/larochelle11a/larochelle11a.pdf). Instead, the experiments in the paper were using a random binarization every time a binary sample was given to the model, and 59600 MNIST examples were used for training (the remaining 400 training examples were used for validation). With this version of the dataset, none of the models exhibited significant amounts of overfitting (up to 0.75 nats difference between test set log likelihood and training set log likelihood). When trained on the 50000 training examples from the fixed binarization of the MNIST dataset, the models overfit significantly more (about 3.5 nats difference between test set log likelihood and training set log likelihood) and achieve test log-likelihoods about 2 nats lower than the ones reported in the paper for the random binarization MNIST. We didn't use any regularization in our experiments to combat the overfitting on the fixed binarization of the MNIST dataset. The VAE models for which the log-likelihood was [reported in the literature](http://arxiv.org/abs/1401.4082) used the fixed binarization dataset augmented by samples with added flip-bit or drop-out noise.