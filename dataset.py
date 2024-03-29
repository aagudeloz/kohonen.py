#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import math
import numpy

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
rms = lambda s: math.sqrt(numpy.sum(s ** 2) / float(s.size))
def rms_normalize(dataset):
    k = rms(dataset)
    return dataset / k

mean = lambda s: numpy.sum(s) / float(s.size)
def mean_normalize(dataset):
    k = mean(dataset)
    return dataset - k

def main(args):
    print >>sys.stderr, "No main defined for this module"
    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
