#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import numpy
import math

class NetworkTopologyTore(object):
    """
    Represents a neural network topology as a tore:
        the node on the left side of the map is a neighbor of the one on the
        right side (same for up and down).
    """
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
        >>> ntt.distance((0, 0), (0, 1))
        1.0
        """
        xx = abs(x - i)
        xx = min(xx, self._height - xx)

        yy = abs(y - j)
        yy = min(yy, self._width - yy)

        return math.sqrt(xx ** 2 + yy ** 2)

    def neighbors_of(self, node_id, distance):
        """
            Return the neighbors within a given distance,
            distance.
                distance < 1: only the given node
                distance c [1, 2[ : its first neighbors
                ...

            @return [(neighbor_id, weight), ...]
              where weight = 1 - (node_distance / distance))
                           => close node: -> 1
                           => distant node: -> 0

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

def main(args):
    print "Some doctesting..."
    import doctest
    i = doctest.testmod()

    if (i.failed > 0):
        return i.failed
    print "Everything ok: %s tests" % i.attempted

if __name__ == "__main__":
    main(sys.argv)
