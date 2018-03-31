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
import sys, os, subprocess
from sklearn.cluster import DBSCAN

from faster_particles.ppn import PPN
from faster_particles.base_net import basenets
from faster_particles import ToydataGenerator
from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator

CLASSES = ('__background__', 'track_edge', 'shower_start', 'track_and_shower')

def filter_points(im_proposals, im_scores):
    db = DBSCAN(eps=9.0, min_samples=1).fit_predict(im_proposals)
    print(db)
    keep = {}
    index = {}
    for i in range(len(db)):
        cluster = db[i]
        if cluster not in keep.keys() or im_scores[i] > keep[cluster]:
            keep[cluster] = im_scores[i]
            index[cluster] = i
    new_proposals = []
    for cluster in keep:
        new_proposals.append(im_proposals[index[cluster]])
    return np.array(new_proposals)

def display(blob, cfg, im_proposals=None, ppn1_proposals=None, ppn1_labels=None,
            rois=None, ppn2_proposals=None, ppn2_positives=None, im_labels=None,
            im_scores=None, ppn2_pixels_pred=None, index=0, dim1=8, dim2=4, name='display'):
    #fig, ax = plt.subplots(1, 1, figsize=(18,18), facecolor='w')
    #ax.imshow(blob['data'][0,:,:,0], interpolation='none', cmap='hot', origin='lower')
    print(im_proposals)
    print(im_scores)
    print(im_labels)

    N = blob['data'].shape[1]
    N2 = int(N/dim1) # F3 size
    N3 = int(N2/dim2) # F5 size

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(blob['data'][0,:,:,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)

    #for gt_pixel in blob['gt_pixels']:
    #    x, y = gt_pixel[1], gt_pixel[0]
    #    if gt_pixel[2] == 1:
    #        plt.plot([x], [y], 'ro')
    #    elif gt_pixel[2] == 2:
    #        plt.plot([x], [y], 'go')
    """if ppn1_proposals is not None:
        for i in range(len(ppn1_proposals)):
            if ppn1_labels is None or ppn1_labels[i] == 1:
                plt.plot([ppn1_proposals[i][1]*32.0], [ppn1_proposals[i][0]*32.0], 'r+')
                coord = np.floor(ppn1_proposals[i])*32.0
                #print(floor(coord[1]), floor(coord[0]))
                ax.add_patch(
                    patches.Rectangle(
                        (coord[1], coord[0]),
                        32, # width
                        32, # height
                        #fill=False,
                        #hatch='\\',
                        facecolor='green',
                        alpha = 0.5,
                        linewidth=1.0,
                        edgecolor='red',
                    )
                )
    """
    if rois is not None:
        for roi in rois:
            #print(roi[1]*32.0, roi[0]*32.0)
            x, y = roi[1], roi[0]
            #if not cfg.TOYDATA:
            #    x = roi[0]
            #    y = roi[1]
            ax.add_patch(
                patches.Rectangle(
                    (x*dim1*dim2, y*dim1*dim2),
                    dim1, # width
                    dim1, # height
                    #fill=False,
                    #hatch='\\',
                    facecolor='pink',
                    alpha = 0.3,
                    linewidth=1.0,
                    edgecolor='black',
                )
            )

        # if ppn2_proposals is not None:
        #for i in range(len(rois)):
        #    if ppn2_positives is None or ppn2_positives[i]:
        #        plt.plot([ppn2_proposals[i][1]*8.0+rois[i][1]*32.0], [ppn2_proposals[i][0]*8.0+rois[i][0]*32.0], 'b+')
        #        coord = np.floor(ppn2_proposals[i])*8.0 + rois[i]*32.0
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

    ax.set_xlim(0, cfg.IMAGE_SIZE)
    ax.set_ylim(0, cfg.IMAGE_SIZE)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_proposals_%d.png' % index))
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(blob['data'][0,:,:,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)

    if im_proposals is not None and im_scores is not None:
        if len(im_proposals) > 0:
            im_proposals = filter_points(im_proposals, im_scores)
        for i in range(len(im_proposals)):
            proposal = im_proposals[i]
            #print(im_labels[i])
            #plt.text(proposal[1], proposal[0], str(im_scores[i][im_labels[i]]))
            x, y = proposal[1], proposal[0]
            #if not cfg.TOYDATA:
            #    x = proposal[0]
            #    y = proposal[1]
            if im_labels[i] == 0: # Track
                plt.plot([x], [y], 'yo')
            elif im_labels[i] == 1: #Shower
                plt.plot([x], [y], 'go')
            else:
                raise Exception("Label unknown")
    ax2.set_xlim(0, cfg.IMAGE_SIZE)
    ax2.set_ylim(0, cfg.IMAGE_SIZE)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_predictions_%d.png' % index))
    plt.close(fig2)

def statistics(cfg, im_proposals, im_labels, im_scores):
    track_scores = []
    shower_scores = []
    for i in range(len(im_scores)):
        if im_labels[i] == 0:
            track_scores.append(im_scores[i])
        else:
            shower_scores.append(im_scores[i])
    bins = np.linspace(0, 1, 20)
    plt.hist(track_scores, bins, alpha=0.5, label='track')
    plt.hist(shower_scores, bins, alpha=0.5, label='shower')
    plt.yscale('log', nonposy='clip')
    plt.legend(loc='upper right')
    plt.xlabel("Score")
    plt.ylabel("#Proposals")
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, 'scores.png'))

def get_filelist(ls_command):
    filelist = subprocess.Popen(["ls %s" % ls_command], shell=True, stdout=subprocess.PIPE).stdout
    return str(filelist.read().splitlines()).replace('\'', '\"').replace(" ", "")

def inference(cfg):
    if cfg.NET == 'ppn':
        if cfg.TOYDATA:
            data = ToydataGenerator(cfg)
        else:
            filelist = get_filelist(cfg.DATA)
            data = LarcvGenerator(cfg, ioname="inference", filelist=filelist)
    else:
        data = ToydataGenerator(cfg, classification=True)

    if cfg.NET == 'ppn':
        net = PPN(cfg=cfg)
    elif cfg.NET == 'base':
        net = basenets[cfg.BASE_NET](cfg=cfg)
    net.init_placeholders()
    net.create_architecture(is_training=False)

    saver = tf.train.Saver()
    im_proposals, im_labels, im_scores = [], [], []
    with tf.Session() as sess:
        saver.restore(sess, cfg.WEIGHTS_FILE)
        for i in range(10):
            blob = data.forward()
            summary, results = net.test_image(sess, blob)
            if cfg.NET == 'ppn':
                #print(blob, results)
                im_proposals.extend(results['im_proposals'])
                im_labels.extend(results['im_labels'])
                im_scores.extend(results['im_scores'])
                display(blob, cfg, index=i, dim1=net.dim1, dim2=net.dim2, **results)
            else:
                print(blob, results)
    statistics(cfg, im_proposals, im_labels, im_scores)


if __name__ == '__main__':
    inference(cfg)
