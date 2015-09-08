import urllib
import gzip

if __name__=='__main__':
	mnist_filenames = ['train-images-idx3-ubyte', 't10k-images-idx3-ubyte']

	for filename in mnist_filenames:
		local_filename = "MNIST/{}".format(filename)
		urllib.urlretrieve ("http://yann.lecun.com/exdb/mnist/{}.gz".format(filename), local_filename+'.gz')
		with gzip.open(local_filename+'.gz', 'rb') as f:
			file_content = f.read()
		with open(local_filename, 'wb') as f:
			f.write(file_content)

	subdatasets = ['train', 'valid', 'test']
	for subdataset in subdatasets:
		filename = 'binarized_mnist_{}.amat'.format(subdataset)
		local_filename = "BinaryMNIST/{}".format(filename)
		urllib.urlretrieve ("http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat", local_filename)