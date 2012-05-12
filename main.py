#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import random
from dataset import mnist_load, mnist_get_image, rms_normalize, mean_normalize
from network import NeuralNetwork
from topology import NetworkTopologyTore
from trainer import NeuralTrainer, distance_scalar, distance_euclidean

# Go!
def main(args):
    random.seed(1)

    print "Some doctesting..."
    import doctest
    i = doctest.testmod()

    if (i.failed > 0):
        return i.failed
    print "Everything ok: %s tests" % i.attempted

    nn = NeuralNetwork(2, 3)
    topology = NetworkTopologyTore(2, 2, 1)
    nt = NeuralTrainer(nn, topology, distance_euclidean)

    def ignore():
        ts = [[0, 0, 1], [0, 0.5, 1]] #, [1, 0, 0], [0.3, 0, 0]]

        print "init"
        print nn._matrix

        print "train"
        for i in range(10):
            nt.train(ts)
        print nt.classify([[0, 0.1, 1.2], [0.2, 0.2, 0.8]])

        print "result"
        print nn._matrix

    print "Load the mnist db..."
    train_set, valid_set, test_set = mnist_load()

    print "Construct the network..."
    size_attr = len(mnist_get_image(train_set, 0))
    size_network = 3 * 3

    nn = NeuralNetwork(size_network, size_attr)
    topology = NetworkTopologyTore(size_network, 3, 3)
    nt = NeuralTrainer(nn, topology, distance_euclidean)

    print "=== Training"
    print "Normalize the training set..."

    ts = [train_set[0][0:30], train_set[1][0:30]][0]
    # Training
    #dataset = [rms_normalize(mean_normalize(s)) for s in tmp_training[0]]
    random.shuffle(ts)

    print "Start the training..."
    nt = NeuralTrainer(nn, topology, distance_scalar)
    nt.train(ts)
    print "Finished..."

    print "=== Classification"
    print "Normalize the test set..."

    print "Start the classification..."

    ids = [i for i in xrange(len(train_set[1][:5000])) if train_set[1][i] == 2]
    ts = [train_set[0][i] for i in ids]
    results = nt.classify(ts)
    print "Finished..."

    print "=== Finally"
    print dict((i,results.count(i)) for i in results)

if __name__ == "__main__":
    main(sys.argv)
