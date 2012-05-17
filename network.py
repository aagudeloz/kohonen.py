#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import numpy

# Neural Network Definition
class NeuralNetwork(object):
    def __init__(self, size_network, size_attr):
        self._size_network = size_network
        self._size_attr = size_attr

        self._matrix = numpy.random.random((size_network, size_attr)) * 0.5

    def __getitem__(self, neuron_id):
        return self._matrix[neuron_id]

    def __setitem__(self, neuron_id, value):
        self._matrix[neuron_id] = value

    def __len__(self):
        return self._size_network

    def __str__(self):
        return str(self._matrix)

def main(args):
    print >>sys.stderr, "No main defined for this module"
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
