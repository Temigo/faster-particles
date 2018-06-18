from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

#ppn1_distances_to_closest_gt = np.genfromtxt(os.path.join(directory, "ppn1_distances_to_closest_gt.csv"), delimiter=",")
#ppn1_distances_to_closest_pred = np.genfromtxt(os.path.join(directory, "ppn1_distances_to_closest_pred.csv"), delimiter=",")
#ppn1_false_positives = np.genfromtxt(os.path.join(directory, "ppn1_false_positives.csv"), delimiter=",")
#ppn1_false_negatives = np.genfromtxt(os.path.join(directory, "ppn1_false_negatives.csv"), delimiter=",")
threshold_false_positive = 5
threshold_false_negative = 5
directory = sys.argv[3]

def make_plot(data_2d, data_3d, bins=None, xlabel="", ylabel="", filename="", line=False, **kwargs):
    plt.hist(data_3d, bins, alpha=1.0, label='3d', color='#084c61', **kwargs)
    #plt.hist(data_2d, bins, alpha=0.8, label='2d', color='#db504a', **kwargs)
    if 'cumulative' in kwargs.keys():
        plt.yscale('log')
        plt.xscale('log')
    if line:
        plt.axhline(y=0.1, color='#e3b505', linestyle='dashed', linewidth=1)
        plt.axhline(y=0.05, color='#e3b505', linestyle='dashed', linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.legend(loc='upper right')
    plt.savefig(os.path.join(directory, filename))
    print("Saved %s" % os.path.join(directory, filename))
    plt.gcf().clear()

def load(directory):
    ppn2_distances_to_closest_gt = np.genfromtxt(os.path.join(directory, "ppn2_distances_to_closest_gt.csv"), delimiter=",")
    ppn2_distances_to_closest_pred = np.genfromtxt(os.path.join(directory, "ppn2_distances_to_closest_pred.csv"), delimiter=",")
    ppn2_false_positives = np.genfromtxt(os.path.join(directory, "ppn2_false_positives.csv"), delimiter=",")
    ppn2_false_negatives = np.genfromtxt(os.path.join(directory, "ppn2_false_negatives.csv"), delimiter=",")
    print(ppn2_distances_to_closest_gt, ppn2_distances_to_closest_pred, ppn2_false_positives, ppn2_false_negatives)
    return ppn2_distances_to_closest_gt, ppn2_distances_to_closest_pred, ppn2_false_positives, ppn2_false_negatives

ppn2_distances_to_closest_gt_2d, ppn2_distances_to_closest_pred_2d, ppn2_false_positives_2d, ppn2_false_negatives_2d = load(sys.argv[1])
ppn2_distances_to_closest_gt_3d, ppn2_distances_to_closest_pred_3d, ppn2_false_positives_3d, ppn2_false_negatives_3d = load(sys.argv[2])

make_plot(ppn2_distances_to_closest_gt_2d, ppn2_distances_to_closest_gt_3d,
    bins=np.linspace(0, 20, 100),
    xlabel="Distance to nearest ground truth pixel",
    ylabel="Proposed pixels",
    filename='ppn2_distance_to_closest_gt.png'
)

make_plot(ppn2_distances_to_closest_gt_2d, ppn2_distances_to_closest_gt_3d,
    bins=100,
    xlabel="Distance to nearest ground truth pixel",
    ylabel="Cumulative fraction of proposed pixels",
    filename='ppn2_distance_to_closest_gt_cumulative.png',
    cumulative=-1,
    density=True,
    histtype='step',
    line=True
)

make_plot(ppn2_distances_to_closest_gt_2d, ppn2_distances_to_closest_gt_3d,
    bins=np.linspace(0, 5, 50),
    xlabel="Distance to nearest ground truth pixel",
    ylabel="Proposed pixels",
    filename='ppn2_distance_to_closest_gt_zoom.png'
)

make_plot(ppn2_distances_to_closest_pred_2d, ppn2_distances_to_closest_pred_3d,
    bins=np.linspace(0, 20, 100),
    xlabel="Distance to nearest proposed pixel",
    ylabel="Ground truth pixels",
    filename='ppn2_distance_to_closest_pred.png'
)

make_plot(ppn2_distances_to_closest_pred_2d, ppn2_distances_to_closest_pred_3d,
    bins=100,
    xlabel="Distance to nearest proposed pixel",
    ylabel="Cumulative fraction of ground truth pixels",
    filename='ppn2_distance_to_closest_pred_cumulative.png',
    cumulative=-1,
    density=True,
    histtype='step',
    line=True
)

make_plot(ppn2_distances_to_closest_pred_2d, ppn2_distances_to_closest_pred_3d,
    bins=np.linspace(0, 5, 50),
    xlabel="Distance to nearest proposed pixel",
    ylabel="Ground truth pixels",
    filename='ppn2_distance_to_closest_pred_zoom.png'
)

# #proposed points at a distance > threshold from any ground truth point / # total proposed point
# this is per image
make_plot(ppn2_false_positives_2d, ppn2_false_positives_3d,
    bins=np.linspace(0, 1, 10),
    xlabel="Percentage of proposed points at a distance > %d from any ground truth point" % threshold_false_positive,
    ylabel="Images",
    filename='ppn2_false_positives.png'
)

# #gt pixels at a distance > threshold from any proposed point / # total gt pixels
# this is per image
make_plot(ppn2_false_negatives_2d, ppn2_false_negatives_3d,
    bins=np.linspace(0, 1, 10),
    xlabel="Percentage of ground truth pixels at a distance > %d from any proposed point" % threshold_false_negative,
    ylabel="Images",
    filename='ppn2_false_negatives.png'
)

def percentage(d):
    perc2d = len(ppn2_distances_to_closest_gt_2d[ppn2_distances_to_closest_gt_2d > d]) / len(ppn2_distances_to_closest_gt_2d) * 100
    print("Percentage of proposed pixels whose distance to nearest gt is > %d pix (2d) = " % d, perc2d)

    perc3d = len(ppn2_distances_to_closest_gt_3d[ppn2_distances_to_closest_gt_3d > d]) / len(ppn2_distances_to_closest_gt_3d) * 100
    print("Percentage of proposed pixels whose distance to nearest gt is > %d pix (3d) = " % d, perc3d)

    perc2d1 = len(ppn2_distances_to_closest_pred_2d[ppn2_distances_to_closest_pred_2d > d]) / len(ppn2_distances_to_closest_pred_2d) * 100
    print("Percentage of gt pixels whose distance to nearest pred is > %d pix (2d) = " % d, perc2d1)

    perc3d1 = len(ppn2_distances_to_closest_pred_3d[ppn2_distances_to_closest_pred_3d > d]) / len(ppn2_distances_to_closest_pred_3d) * 100
    print("Percentage of gt pixels whose distance to nearest pred is > %d pix (3d) = " % d, perc3d1)

    mean2d = np.mean(ppn2_distances_to_closest_gt_2d)
    mean3d = np.mean(ppn2_distances_to_closest_gt_3d)
    print("Mean of ppn2_distances_to_closest_gt_2d = ", mean2d)
    print("Mean of ppn2_distances_to_closest_gt_3d = ", mean3d)
percentage(5.0)
percentage(10.0)
percentage(15.0)
percentage(20.0)
