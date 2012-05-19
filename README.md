Kohonen.py
==========

Kohonen neural network implementation in Python.


Dependencies
------------

* python >= 2.6
* numpy
* matplotlib


Usage
-----

    python main.py mnist_path valid_count test_count train_count network_width

* mnist_path: the [cpickled mnist database](http://deeplearning.net/data/mnist/mnist.pkl.gz)
* valid_count: the size of the validation dataset (<= 10000)
* test_count: the size of the testing dataset (<= 10000)
* train_count: the size of the training dataset (<= 60000)
* network_width: the size of the neural network (between 4 and 30 is good)

For example:

    python main.py data/mnist.pkl.gz 100 100 600 10

is a good starting point

More info on [while1read.com](http://www.while1read.com/kohonen-py/).
