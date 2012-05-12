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
