import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from multichannel_utils import produce_graphs, examine_proposals, delete_proposals
from threshold_utils import produce_graphs_overlayed

# what can vary? score_threshold which produced the data sets, distance_threshold will affect the filtering, 
# do not change num or coord1 or coord2 unless new data set is generated, change arguments of produce_graph 
# at the bottom to change which channel's statistics are shown
score_threshold = 0.6
distance_threshold_array = [0.2, 0.5, 1, 2, 5, 10, 20]
num = 10000 # same as first dimension of any of the np arrays
coord1 = 0 #shared coordinate is first index in first array
coord2 = 1 #shared coordinate is second index in second array
proposals_all = []
truth_all = []
directory = './multichannel/demo_' + str(score_threshold) + '/'

for distance_threshold in distance_threshold_array: #a proposal farther than this from any proposal in another channel is removed
    #TODO: make more object-oriented... would reduce lines of code by half probably
    channel1 = str(score_threshold)
    display_dir1 = '../display/demo_' + channel1 + '/metrics'
    labels1 = np.load(display_dir1 + '/ppn2_labels.npy') # 10000 by (numproposals) by 1
    proposals1 = np.load(display_dir1 + '/ppn2_proposals.npy') # 10000 by (numproposals) by 2
    scores1 = np.load(display_dir1 + '/ppn2_scores.npy') # 10000 by (numproposals) by 1
    truth1 = np.load(display_dir1 + '/ppn2_truth.npy') # 10000 by (numproposals) by 1

    channel2 = str(score_threshold) + '_channel'
    display_dir2 = '../display/demo_' + channel2 + '/metrics'
    labels2 = np.load(display_dir2 + '/ppn2_labels.npy') # 10000 by (numproposals) by 1
    proposals2 = np.load(display_dir2 + '/ppn2_proposals.npy') # 10000 by (numproposals) by 2
    scores2 = np.load(display_dir2 + '/ppn2_scores.npy') # 10000 by (numproposals) by 1
    truth2 = np.load(display_dir2 + '/ppn2_truth.npy') # 10000 by (numproposals) by 1


    prune_indices1 = examine_proposals(proposals1, proposals2, num, distance_threshold, coord1, coord2)
    prune_indices2 = examine_proposals(proposals2, proposals1, num, distance_threshold, coord2, coord1)

    #proposal, score, label should be modified by multichannel analysis. truth should not.
    proposals1, scores1, labels1 = delete_proposals(proposals1, scores1, labels1, prune_indices1, num)
    proposals2, scores2, labels2 = delete_proposals(proposals2, scores2, labels2, prune_indices2, num)
    
    proposals_all.append(proposals1)
    truth_all.append(truth1)
    
produce_graphs_overlayed(proposals_all, truth_all, num, distance_threshold_array, directory)
