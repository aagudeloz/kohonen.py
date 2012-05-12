#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import numpy
import math

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

def main(args):
    print >>sys.stderr, "No main defined for this module"
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
