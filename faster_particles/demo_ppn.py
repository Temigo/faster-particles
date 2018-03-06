# *-* encoding: utf-8 *-*
# Demo for PPN
# Usage: python demo_ppn.py model.ckpt
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import floor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import sys

from ppn import PPN
from faster_particles import ToydataGenerator

CLASSES = ('__background__', 'track_edge', 'shower_start', 'track_and_shower')

def display(blob, im_proposals=None, ppn1_proposals=None, ppn1_labels=None,
            rois=None, ppn2_proposals=None, ppn2_positives=None, im_labels=None,
            im_scores=None, index=0, name='display'):
    #fig, ax = plt.subplots(1, 1, figsize=(18,18), facecolor='w')
    #ax.imshow(blob['data'][0,:,:,0], interpolation='none', cmap='hot', origin='lower')
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(blob['data'][0,:,:,0], cmap='coolwarm', interpolation='none', origin='lower')
    for gt_pixel in blob['gt_pixels']:
        if gt_pixel[2] == 1:
            plt.plot([gt_pixel[1]], [gt_pixel[0]], 'ro')
        elif gt_pixel[2] == 2:
            plt.plot([gt_pixel[1]], [gt_pixel[0]], 'go')
    if ppn1_proposals is not None:
        for i in range(len(ppn1_proposals)):
            if ppn1_labels is None or ppn1_labels[i] == 1:
                plt.plot([ppn1_proposals[i][1]*32.0], [ppn1_proposals[i][0]*32.0], 'y+')
                coord = np.floor(ppn1_proposals[i])*32.0
                #print(floor(coord[1]), floor(coord[0]))
                ax.add_patch(
                    patches.Rectangle(
                        (coord[1], coord[0]),
                        8, # width
                        8, # height
                        #fill=False,
                        #hatch='\\',
                        facecolor='green',
                        alpha = 0.5,
                        linewidth=1.0,
                        edgecolor='red',
                    )
                )

    if rois is not None and ppn2_proposals is not None:
        for roi in rois:
            #print(roi[1]*32.0, roi[0]*32.0)
            ax.add_patch(
                patches.Rectangle(
                    (roi[1]*32.0, roi[0]*32.0),
                    8, # width
                    8, # height
                    #fill=False,
                    #hatch='\\',
                    facecolor='pink',
                    alpha = 0.3,
                    linewidth=1.0,
                    edgecolor='black',
                )
            )

        for i in range(len(rois)):
            if ppn2_positives is None or ppn2_positives[i]:
                plt.plot([ppn2_proposals[i][1]*8.0+rois[i][1]*32.0], [ppn2_proposals[i][0]*8.0+rois[i][0]*32.0], 'b+')
                coord = np.floor(ppn2_proposals[i])*8.0 + rois[i]*32.0
                #print(floor(coord[1]), floor(coord[0]))
                """ax.add_patch(
                    patches.Rectangle(
                        (coord[1], coord[0]),
                        8, # width
                        8, # height
                        #fill=False,
                        #hatch='\\',
                        facecolor='yellow',
                        alpha = 0.5,
                        linewidth=1.0,
                        edgecolor='pink',
                    )
                )"""

    if im_proposals is not None and im_scores is not None:
        for i in range(len(im_proposals)):
            proposal = im_proposals[i]
            print(im_labels[i])
            plt.text(proposal[1], proposal[0], str(im_scores[i][im_labels[i]]))
            if im_labels[i] == 0: # Track
                plt.plot([proposal[1]], [proposal[0]], 'y+')
            elif im_labels[i] == 1: #Shower
                plt.plot([proposal[1]], [proposal[0]], 'y*')
            else:
                raise Exception("Label unknown")

    #plt.imsave('display.png', blob['data'][0,:,:,0])
    plt.savefig('display/' + name + '%d.png' % index)
    plt.close(fig)

def inference(cfg):
    toydata = ToydataGenerator(cfg)

    net = PPN()
    net.init_placeholders()
    net.create_architecture(is_training=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, cfg.WEIGHTS_FILE)
        for i in range(10):
            blob = toydata.forward()
            im_proposals, im_labels, im_scores, ppn1_proposals, rois, ppn2_proposals = net.test_image(sess, blob)
            display(blob, im_proposals=im_proposals, im_labels=im_labels, im_scores=im_scores, index=i)

if __name__ == '__main__':
    inference(cfg)
