#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import numpy
import math
import random

# mnist manipulation
mnist_get_image = lambda mset, data_id: mset[0][data_id]
mnist_get_label = lambda mset, data_id: mset[1][data_id]

def mnist_load(path="data/mnist.pk1.gz"):
    """
        Load the mnist dataset,
        from: http://deeplearning.net/tutorial/gettingstarted.html

        a set is defined as:
            [images][labels]
            where images = [[pixel0 - pixel784 (28 * 28)], [...], [...]]
                  labels = [label1, label2,...]

        @param path the mnist file path (default = data/mnist.pk1.gz)
        @return (train_set, valid_set, test_set)
    """
    import cPickle, gzip

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    return train_set, valid_set, test_set

# RMS computation
rms = lambda s: math.sqrt(sum(map(lambda x: x ** 2, s)) / len(s))
def rms_normalize(dataset):
    k = rms(dataset)
    return map(lambda x: x / k, dataset)

mean = lambda s: sum(s) / float(len(s))
def mean_normalize(dataset):
    k = mean(dataset)
    return map(lambda x: x - k, dataset)


# Sum definitions
def distance_scalar(data, neuron):
    """
    1 - {data} . {neuron}
     (data and neuron should both have 0 mean and rms = 1)
    >>> distance_scalar([2, 45], [235, 988]) == \
        distance_scalar([235, 988], [2, 45])
    True
    >>> distance_scalar([0, 1], [1, 0])
    1
    >>> distance_scalar([0, 1], [0, 1])
    0
    """
    try:
        return 1 - sum(map(lambda x, y: x * y, data, neuron))
    except ZeroDivisionError:
        return float("inf")

def distance_euclidean(data, neuron):
    """
    >>> distance_euclidean([0, 1], [1, 0]) == math.sqrt(2)
    True
    >>> distance_euclidean([0, 1], [0, 1])
    0.0
    >>> distance_euclidean([0, 5], [0, 1])
    4.0
    """
    return math.sqrt(sum(map(lambda x, y: (x - y) ** 2, data, neuron)))


# Neural Network Definition
class NeuralNetwork(object):
    def __init__(self, size_network, size_attr):
        self._size_network = size_network
        self._size_attr = size_attr

        self._matrix = numpy.random.random((size_network, size_attr))

    def __getitem__(self, neuron_id):
        return self._matrix[neuron_id]

    def __setitem__(self, neuron_id, value):
        self._matrix[neuron_id] = value

    def __len__(self):
        return self._size_network

    def __str__(self):
        return str(self._matrix)

class NeuralTrainer(object):
    def __init__(self, network, topology, distance_function):
        self._network = network
        self._topology = topology
        self._distf = distance_function

    def pick_nearest(self, data):
        dfunc = lambda x: self._distf(x, data) # Sugar
        dists = [dfunc(self._network[i]) for i in range(len(self._network))]

        return dists.index(min(dists))

    def train(self, dataset):
        """
        Training:
            for each data, we pick the closest neuron (the winner)
                We adapt its weight to be closer to the data
                    and we also adapt its neighbors
            TODO: Adapt the neighbors
            TODO: "The conscience"
        """
        radius = len(self._topology)

        # simulate FP's r/w closure, I'm still looking for a better solution
        # TODO: move this definition outside
        def learning_rate(current=[1]):
            r = current[0]
            current[0] /= 2.0
            return r

        def radius_rate(current=[len(self._topology)]):
            r = current[0]
            current[0] /= 1.2
            return r

        for d in dataset:
            winner = self.pick_nearest(d)

            print "=" * 50
            print "winner:", winner
            print "neighbors:", self._topology.neighbors_of(winner, radius)

            # Thanks to the neighbors definition,
            # winner is now a simple special case :)
            winners = self._topology.neighbors_of(winner, radius)

            # Adapt the winners
            for (winner, meaningful) in winners:
                weights = self._network[winner]
                weights += meaningful * learning_rate() * (numpy.array(d) - weights)
                print "m: %s, l: %s, sub: %s" % (meaningful, learning_rate(),
                                                  (numpy.array(d) - weights))
                #TODO: remove this numpy.array

    def classify(self, dataset):
        return map(self.pick_nearest, dataset)

class NetworkTopologyTore(object):
    def __init__(self, network_size, height, width):
        # Care: no size checking, numpy might raise an error!

        self._height = height
        self._width = width
        self._matrix = numpy.arange(network_size).reshape(height, width)

    def __len__(self):
        """
        >>> ntt = NetworkTopologyTore(9, 3, 3)
        >>> len(ntt)
        4
        """
        return math.sqrt(self._height ** 2 + self._width ** 2)

    def distance(self, (x, y), (i, j)):
        """
        Euclidian distance on a torus

        >>> ntt = NetworkTopologyTore(9, 3, 3)
        >>> ntt.distance((1, 1), (1, 2))
        1.0
        >>> ntt.distance((4, 4), (4, 4))
        0.0
        >>> ntt.distance((1, 1), (1, 4))
        0.0
        """
        xx = abs(x - i)
        xx = min(xx, self._width - xx)

        yy = abs(y - j)
        yy = min(yy, self._height - yy)

        return math.sqrt(xx ** 2 + yy ** 2)

    def neighbors_of(self, node_id, distance):
        """
            Return the neighbors within a given distance,
            distance.
                distance < 1: only the given node
                distance c [1, 2[ : its first neighbors (4 connectivity)

            @return [(neighbor_id, weight), ...]
              where weight = 1 - (node_distance / distance))

            >>> ntt = NetworkTopologyTore(9, 3, 3)
            >>> ntt.neighbors_of(0, 0) == set((0, 1.0))
            True
            >>> ntt.neighbors_of(0, 1) == \
                    set([(0, 1.0), (1, 0.0), (3, 0.0), (2, 0.0), (6, 0.0)])
            True
        """

        if (distance <= 0):
            return set((node_id, 1.0))

        # Get the x, y coordinates for the given node
        f = abs(self._matrix - node_id).argmin()
        x, y = divmod(f, self._width)

        results = set()

        # Dummy search in the whole set
        for i in xrange(self._height):
            for j in xrange(self._width):
                d = self.distance((x, y), (i, j))
                if (d <= distance):
                    results.add((self._matrix[i, j], 1 - (d / distance)))

        return results

# Go!
def main(args):
    print "Some doctesting..."
    import doctest
    i = doctest.testmod()

    if (i.failed > 0):
        return i.failed
    print "Everything ok: %s tests" % i.attempted

    print "Load the mnist db..."
    train_set, valid_set, test_set = mnist_load()

    print "Construct the network..."
    size_attr = len(mnist_get_image(train_set, 0))
    size_network = 3 * 3

    nn = NeuralNetwork(size_network, size_attr)
    topology = NetworkTopologyTore(size_network, 3, 3)

    print "=== Training"
    print "Normalize the training set..."

    tmp_training = [train_set[0][0:30], train_set[1][0:30]]
    # Training
    dataset = [rms_normalize(mean_normalize(s)) for s in tmp_training[0]]
    random.shuffle(dataset)

    print "Start the training..."
    nt = NeuralTrainer(nn, topology, distance_scalar)
    nt.train(dataset)
    print "Finished..."

    print "=== Classification"
    print "Normalize the test set..."

    print "Start the classification..."
    results = nt.classify(dataset)
    print "Finished..."

    print "=== Finally"
    print results

if __name__ == "__main__":
    main(sys.argv)
