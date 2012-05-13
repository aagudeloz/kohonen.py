#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import random
from dataset import mnist_load, mnist_get_image, rms_normalize, mean_normalize
from network import NeuralNetwork
from topology import NetworkTopologyTore
from trainer import NeuralTrainer, distance_scalar, distance_euclidean

import pylab as pl
import matplotlib as mpl
import colorsys

def plot(results):
    """
        Plot the results
        @param results {numberA: { neuronX: 12, neuronY: 13, ...}, ...}
    """
    fig = pl.figure()

    legend = [[], []] # [[color1, color2,...], [id1, id2,...]]
    val_bottom = [0] * 11#len(results) # The current height of the chart

    for (k, v) in results.items():
        img_id = k
        neurons = v.keys()
        values = v.values()

        color = colorsys.hsv_to_rgb(img_id / 10.0, 1, 0.5)
        bar_bottom = [val_bottom[n] for n in neurons]

        subplot = fig.add_subplot(1, 1, 1)
        sb = subplot.bar(neurons, values, align="center", facecolor=color, bottom=bar_bottom)

        legend[0].append(sb[0])
        legend[1].append(img_id)

        # We move the bottom coordinate for the stacked bars
        for i in xrange(len(neurons)):
            val_bottom[neurons[i]] += values[i]

    subplot.legend(*legend) # lazy solution
    subplot.set_ylabel("matching count")
    subplot.set_title("Neurons match per number")

    fig.savefig("result.png")
#    pl.show()

def plot_network(network, filepath):
    print "oula"
    f = pl.figure()

    for i in xrange(len(network)):
        s = f.add_subplot(5, 5, i + 1)
        s.matshow(network[i].reshape(28, 28))
        s.set_title("network %s" % i)
        s.axis("off")
    f.savefig(filepath)

# Go!
def main(args):
    random.seed(1001)

    print "Some doctesting..."
    import doctest
    i = doctest.testmod()

    if (i.failed > 0):
        return i.failed
    print "Everything ok: %s tests" % i.attempted

    def ignore():
        nn = NeuralNetwork(2, 3)
        topology = NetworkTopologyTore(2, 2, 1)
        nt = NeuralTrainer(nn, topology, distance_euclidean)

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
    size_network = 3 * 4

    nn = NeuralNetwork(size_network, size_attr)
    topology = NetworkTopologyTore(size_network, 3, 4)
    nt = NeuralTrainer(nn, topology, distance_euclidean)

    print "=== Training"
    print "Normalize the training set..."

    ts = [train_set[0][0:500], train_set[1][0:500]][0]
    # Training
    #dataset = [rms_normalize(mean_normalize(s)) for s in tmp_training[0]]
    random.shuffle(ts)

    print "Start the training..."
    nt = NeuralTrainer(nn, topology, distance_scalar)

    def inter_tick(network, trainer, accu=[0]):
        if accu[0] % 100 == 1:
            plot_network(network, "network-%s.png" % accu[0])
        accu[0] += 1

    nt.train(ts,update_fct=inter_tick)
    print "Finished..."

    print "=== Classification"
    print "Normalize the test set..."

    print "Start the classification..."
    ids = train_set[1][:100]
    ts = train_set[0][:100]
    results = nt.classify(ts)
    print "Finished..."

    print "=== Finally"
    # Build the dict {numberA: { neuronX: 12, neuronY: 13, ...}, ...}
    plottable = dict((i, {}) for i in xrange(10))

    for i in xrange(len(ts)):
        img_id = ids[i]
        neuron = results[i]
        img_idtoneurons = plottable[img_id]
        img_idtoneurons[neuron] = img_idtoneurons.get(neuron, 0) + 1

    print "plot..."
    print "=" * 10
    print results
    print "=" * 10
    print plottable
    print "=" * 10
    plot(plottable)

if __name__ == "__main__":
    main(sys.argv)
