import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from threshold_utils import produce_graphs_overlayed

num = 10000 # same as first dimension of any of the np arrays
score_array = [0.4, 0.6, 0.8]
proposals_all = []
truth_all = []
directory = './threshold_test/'

for score_threshold in score_array:
    channel = str(score_threshold)
    display_dir = '../display/demo_' + channel + '/metrics'
    labels = np.load(display_dir + '/ppn2_labels.npy') # 10000 by (numproposals) by 1
    proposals = np.load(display_dir + '/ppn2_proposals.npy') # 10000 by (numproposals) by 2
    scores = np.load(display_dir + '/ppn2_scores.npy') # 10000 by (numproposals) by 1
    truth = np.load(display_dir + '/ppn2_truth.npy') # 10000 by (numproposals) by 1
    
    proposals_all.append(proposals)
    truth_all.append(truth)

produce_graphs_overlayed(proposals_all, truth_all, num, score_array, directory)
    