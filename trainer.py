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
    return math.sqrt(numpy.sum(numpy.power((data - neuron), 2)))

class NeuralTrainer(object):
    def __init__(self, network, topology, distance_function):
        self._network = network
        self._topology = topology
        self._distf = distance_function

    def pick_nearests(self, data):
        return [self._distf(data, self._network[i]) for i in range(len(self._network))]

    def pick_nearest(self, data, nth=0):
        dists = [self._distf(data, self._network[i]) for i in range(len(self._network))]

        while nth > 0:
            dists.index(min(dists)) == float("inf")
            nth -= 1

        return dists.index(min(dists))

    def pick_match(self, data):
        return [self._distf(data, self._network[i]) for i in range(len(self._network))]

    def train(self, dataset,update_fct=None):
        """
        Training:
            for each data, we pick the closest neuron (the winner)
                We adapt its weight to be closer to the data
                    and we also adapt its neighbors
            TODO: Adapt the neighbors
            TODO: "The conscience"
        """
        topo_radius = len(self._topology) * 0.5

        phi = 0.3
        k =  topo_radius * 1.4 * (1 / (phi * math.sqrt(math.pi * 2)))

        # TODO: parametrize this
        learning_rate = lambda x : ((1 - 1.1 * x) ** 2)
        radius_rate = lambda x : k * math.exp(-0.5 * ((x/phi) ** 2))
        radius_rate = lambda x : (1 - 1 * x) * (topo_radius + 1)

        x = 0
        x_step = 1.0 / len(dataset)

        last_winner = -1
        last_count = 0

        i = 0
        for d in dataset:
            if update_fct is not None:
                update_fct(self._network, self)

            winner = self.pick_nearest(d)

            if winner == last_winner:
                last_count += 1

            if last_count > 5:
                winner = self.pick_nearest(d, 2)
            else:
                last_winner = winner
                last_count = 0

            # Thanks to the neighbors definition,
            # winner is now a simple special case :)
            winners = self._topology.neighbors_of(winner, radius_rate(x))

            #if len(winners) == 1:
            #    break

            l_rate = learning_rate(x)

            print "[tick %s, %s winners]" % (round(x, 2), len(winners)),

            # Adapt the winners
            for (winner, meaningful) in winners:
                weights = self._network[winner]
                weights += meaningful * meaningful *  l_rate * (d - weights)

            x += x_step
            i += 1

    def classify(self, dataset):
        return map(self.pick_nearest, dataset)

    def identify(self, dataset, neuron_map):
        results = []
        for d in dataset:
            vote = {}
            matches = self.pick_match(d)
            for i in range(len(matches)):
                vote[neuron_map[i]] = vote.get(neuron_map[i], 0) + matches[i]
            results.append(max(vote.items(), key=lambda x : x[1])[0])

        return results


def main(args):
    print >>sys.stderr, "No main defined for this module"
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
