#! /usr/bin/python2.6
# -*- coding: utf-8 -*-
import sys
import random
from dataset import mnist_load, mnist_get_image
from network import NeuralNetwork
from topology import NetworkTopologyTore
from trainer import NeuralTrainer, distance_euclidean
import plot

# Go!
def main(args):
    print "python main.py mnist_db_path valid_count test_count train_count network_width"

    mnist_db_path = args[1]
    valid_count = int(args[2])
    test_count = int(args[3])
    train_count = int(args[4])
    network_width = int(args[5])

    random.seed(1001)

    # Init
    # ====

    print "load the mnist db..."
    train_set, valid_set, test_set = mnist_load(mnist_db_path)

    print "filter elements..."
    def filter_set(tset, elements=(0, 1, 2, 3, 4)):
        tmp = [[], []]
        for i in xrange(len(tset[0])):
            if tset[1][i] in elements:
                tmp[0].append(tset[0][i])
                tmp[1].append(tset[1][i])
        return tmp

    #train_set = filter_set(train_set)
    #valid_set = filter_set(valid_set)
    #test_set = filter_set(test_set)

    print "construct the network..."
    size_attr = len(mnist_get_image(train_set, 0))
    network_size = network_width ** 2

    nn = NeuralNetwork(network_size, size_attr)
    topology = NetworkTopologyTore(network_size, network_width, network_width)

    print "=== Training"

    ts = train_set[0][:train_count]
    ids = train_set[1][:train_count]
    random.shuffle(ts)

    print "start the training..."
    nt = NeuralTrainer(nn, topology, distance_euclidean)

    def inner_gen(prefix):
        def inner_tick(network, trainer, accu=[0]):
            if accu[0] < 10 or accu[0] % 40 in (1, 2, 3):
                plot.network_grid(network,
                                  "network-%s%s.png" % (prefix, accu[0]),
                                  (network_width, network_width))
            accu[0] += 1
        return inner_tick

    # Train
    # =====
    nt.train(ts, update_fct=None)#inner_gen("test"))

    print "plot the network..."
    plot.network_grid(nn, "network.png", (network_width, network_width))

    print "generate the classification table..."

    ts = valid_set[0][:valid_count]
    ids = valid_set[1][:valid_count]

    print "classify the training set"
    results = nt.classify(ts)
    neuron_dict = [{} for i in xrange(len(nn))]

    print "generate the table"
    # [{labeli: counti, label:countj}, ...]
    for i in xrange(len(results)):
        neuron = results[i]
        label = ids[i]
        neuron_dict[neuron][label] = neuron_dict[neuron].get(label, 0) + 1

    neuron_map = [-1] * len(nn)
    print len(neuron_map)
    for i in xrange(len(neuron_map)):
        if len(neuron_dict[i].items()) > 0:
            neuron_map[i] = max(neuron_dict[i].items(), key=lambda x: x[1])[0]

    print "plot neuron map"
    plot.neuron_map(topology,
                    neuron_map,
                    "neuron_map.png",
                    (network_width, network_width))

    print "validate the network... (TODO)"
    print "finished..."


    print "=== Classification"
    # print "normalize the test set..."

    ids = test_set[1][:test_count]
    ts = test_set[0][:test_count]

    print "classify..."
    results = nt.classify(ts)

    # Compute the classification rate
    good_classif = 0.0
    for i in xrange(len(results)):
        if neuron_map[results[i]] == ids[i]:
            good_classif += 1
    good_classif /= float(len(results))
    print "on train=%s, test=%s, good classification: %s%%" % \
            (train_count, test_count, round(good_classif, 4)*100)


    print "=== Finally"
    # Build the dict {numberA: { neuronX: 12, neuronY: 13, ...}, ...}
    plottable = dict((i, {}) for i in xrange(10))

    for i in xrange(len(ids)):
        img_id = ids[i]
        neuron = results[i]
        img_idtoneurons = plottable[img_id]
        img_idtoneurons[neuron] = img_idtoneurons.get(neuron, 0) + 1

    print "plot..."

    print "bar"
    plot.classif_bar(plottable, "classifications.png")

    print "matrix count"
    plot.classif_matrix(topology,
                        plottable,
                        "match_count.png",
                        (network_width, network_width))




if __name__ == "__main__":
    #import hotshot
    #prof = hotshot.Profile("test.prof")
    #prof.start()
    main(sys.argv)
    #prof.stop()
    #prof.close()
