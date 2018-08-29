from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.spatial

def produce_graphs_overlayed(proposals_all, truth_all, num, array, directory):
    num_thresholds = len(array)
    ppn2_distances_to_closest_gt = [[] for elem in range(num_thresholds)]
    ppn2_distances_to_closest_pred = [[] for elem in range(num_thresholds)]
    for elem in range(num_thresholds):
        for i in range(num):
            distances_ppn2 = scipy.spatial.distance.cdist(proposals_all[elem][i], truth_all[elem][i])
            try:
                ppn2_distances_to_closest_gt[elem].extend(np.amin(distances_ppn2, axis=1))
            except ValueError:
                pass
            try:
                ppn2_distances_to_closest_pred[elem].extend(np.amin(distances_ppn2, axis=0))
            except ValueError:
                pass
    plot_graphs(ppn2_distances_to_closest_gt, ppn2_distances_to_closest_pred, array, directory)

def plot_graphs(gt, pred, array, directory):
    bins = np.linspace(0, 100, 100)
    make_plot_log(
        gt,
        array, 
        directory,
        bins=50,
        xlabel="distance to nearest ground truth pixel",
        ylabel="#proposed pixels",
        filename='ppn2_distance_to_closest_gt_log.png'
    )
    make_plot_log(
        gt,
        array,
        directory,
        bins=np.linspace(0, 10, 50),
        xlabel="distance to nearest ground truth pixel",
        ylabel="#proposed pixels",
        filename='ppn2_distance_to_closest_gt_zoom_log.png'
    )
    bins = np.linspace(0, 100, 100)
    make_plot_log(
        pred,
        array,
        directory,
        bins=50,
        xlabel="distance to nearest proposed pixel",
        ylabel="#ground truth pixels",
        filename='ppn2_distance_to_closest_pred_log.png'
    )
    make_plot_log(
        pred,
        array,
        directory,
        bins=np.linspace(0, 10, 50),
        xlabel="distance to nearest proposed pixel",
        ylabel="#ground truth pixels",
        filename='ppn2_distance_to_closest_pred_zoom_log.png'
    )

def make_plot_log(data, array, directory, bins=None, xlabel="", ylabel="", filename=""):
    """
    If bins is None: discrete histogram
    """  
    # to determine binning, hopefully representative
    sample_data = np.array(data[0])
    if bins is None:
        d = np.diff(np.unique(sample_data)).min()
        left_of_first_bin = sample_data.min() - float(d)/2
        right_of_last_bin = sample_data.max() + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

    for i in range(len(data)):
        datum = np.array(data[i])
        # modify label depending on what is being varied
        plt.hist(datum, bins, histtype = 'step', label = str(array[i]))
        # plt.hist(datum, bins, label = str(array[i]))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.legend()
    if not os.path.isdir(directory):
        os.makedirs(directory)
    plt.savefig(directory + filename)
    plt.gcf().clear()
        
        