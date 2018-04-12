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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import sys, os, subprocess
from sklearn.cluster import DBSCAN

from faster_particles.ppn import PPN
from faster_particles.base_net import basenets
from faster_particles import ToydataGenerator
from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator

CLASSES = ('__background__', 'track_edge', 'shower_start', 'track_and_shower')

def draw_voxel(x, y, z, size, ax, alpha=0.3, facecolors='pink', **kwargs):
    vertices = [
        [[x, y, z], [x+size, y, z], [x, y+size, z], [x+size, y+size, z]],
        [[x, y, z+size], [x+size, y, z+size], [x, y+size, z+size], [x+size, y+size, z+size]],
        [[x, y, z], [x, y+size, z], [x, y, z+size], [x, y+size, z+size]],
        [[x+size, y, z], [x+size, y+size, z], [x+size, y, z+size], [x+size, y+size, z+size]],
        [[x, y, z], [x+size, y, z], [x, y, z+size], [x+size, y, z+size]],
        [[x, y+size, z], [x+size, y+size, z], [x, y+size, z+size], [x+size, y+size, z+size]]
    ]
    poly = Poly3DCollection(
        vertices,
        **kwargs
    )
    # Bug in Matplotlib with transparency of Poly3DCollection
    # see https://github.com/matplotlib/matplotlib/issues/10237
    poly.set_alpha(alpha)
    poly.set_facecolor(facecolors)
    ax.add_collection3d(poly)

def filter_points(im_proposals, im_scores):
    db = DBSCAN(eps=9.0, min_samples=1).fit_predict(im_proposals)
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

def display(blob, cfg, im_proposals=None, rois=None, im_labels=None, im_scores=None,
            index=0, dim1=8, dim2=4, name='display'):
    #print(im_proposals)
    #print(im_scores)
    #print(im_labels)

    N = blob['data'].shape[1]
    N2 = int(N/dim1) # F3 size
    N3 = int(N2/dim2) # F5 size
    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', **kwargs)

    # Display original image
    if cfg.DATA_3D:
        for voxel in blob['voxels']:
            draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='blue', alpha=0.3, linewidths=0.1, edgecolors='black')
    else:
        ax.imshow(blob['data'][0,...,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)

    # display gt pixels
    if cfg.DATA_3D:
        for gt_pixel in blob['gt_pixels']:
            x, y, z = gt_pixel[2], gt_pixel[1], gt_pixel[0]
            draw_voxel(x, y, z, 1, ax, facecolors='red', alpha=1.0, linewidths=0.3, edgecolors='red')
    else:
        for gt_pixel in blob['gt_pixels']:
            x, y = gt_pixel[1], gt_pixel[0]
            if gt_pixel[2] == 1:
                plt.plot([x], [y], 'ro')
            elif gt_pixel[2] == 2:
                plt.plot([x], [y], 'go')


    if rois is not None:
        for roi in rois:
            if cfg.DATA_3D:
                x, y, z = roi[2], roi[1], roi[0]
                x, y, z = x*dim1*dim2, y*dim2*dim1, z*dim1*dim2
                size = dim1
                draw_voxel(x, y, z, size, ax,
                    facecolors='pink',
                    linewidths=0.01,
                    edgecolors='black',
                    alpha=0.1)
            else:
                x, y = roi[1], roi[0]
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

    ax.set_xlim(0, cfg.IMAGE_SIZE)
    ax.set_ylim(0, cfg.IMAGE_SIZE)
    if cfg.DATA_3D:
        ax.set_zlim(0, cfg.IMAGE_SIZE)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_proposals_%d.png' % index))
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
    #ax2.imshow(blob['data'][0,...,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)
    #ax2.voxels(blob['voxels'], facecolors='blue')

    # Display original image
    if cfg.DATA_3D:
        for voxel in blob['voxels']:
            draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax2, facecolors='blue', alpha=0.3, linewidths=0.1, edgecolors='black')
    else:
        ax2.imshow(blob['data'][0,...,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)

    if im_proposals is not None and im_scores is not None:
        if len(im_proposals) > 0:
            im_proposals = filter_points(im_proposals, im_scores)
        for i in range(len(im_proposals)):
            proposal = im_proposals[i]
            #plt.text(proposal[1], proposal[0], str(im_scores[i][im_labels[i]]))
            if cfg.DATA_3D:
                x, y, z = proposal[2], proposal[1], proposal[0]
                if im_labels[i] == 0: # track
                    ax2.scatter([x], [y], [z], c='yellow')
                elif im_labels[i] == 1: #shower
                    ax2.scatter([x], [y], [z], c='green')
                else:
                    raise Exception("Label unknown")
            else:
                x, y = proposal[1], proposal[0]
                if im_labels[i] == 0: # Track
                    plt.plot([x], [y], 'yo')
                elif im_labels[i] == 1: #Shower
                    plt.plot([x], [y], 'go')
                else:
                    raise Exception("Label unknown")
    ax2.set_xlim(0, cfg.IMAGE_SIZE)
    ax2.set_ylim(0, cfg.IMAGE_SIZE)
    if cfg.DATA_3D:
        ax2.set_zlim(0, cfg.IMAGE_SIZE)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_predictions_%d.png' % index))
    plt.close(fig2)
    print(im_proposals)
    return im_proposals

    """if cfg.DATA_3D and im_proposals is not None and im_scores is not None:
        gt = len(blob['gt_pixels']) # number of gt pixels
        fig3 = plt.figure(figsize=(20, 20*gt))
        fig3, subplots = plt.subplots(gt, 3)
        for j in range(gt):
            gt_pixel = blob['gt_pixels'][j]
            #ax_x = fig3.add_subplot(gt, 3, 3*j+1, aspect='auto')
            #ax_y = fig3.add_subplot(gt, 3, 3*j+2, aspect='auto')
            #ax_z = fig3.add_subplot(gt, 3, 3*j+3, aspect='auto')
            ax_x = subplots[j][0]
            ax_y = subplots[j][1]
            ax_z = subplots[j][2]
            x, y, z = int(gt_pixel[2]), int(gt_pixel[1]), int(gt_pixel[0])
            ax_x.imshow(blob['data'][0,x,:,:,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)
            ax_y.imshow(blob['data'][0,:,y,:,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)
            ax_z.imshow(blob['data'][0,:,:,z,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=400)
            for i in range(len(im_proposals)):
                proposal = im_proposals[i]
                xi, yi, zi = proposal[2], proposal[1], proposal[0]
                if im_labels[i] == 0: # Track
                    ax_x.scatter(yi, zi, c='yellow', marker='o', s=0.5)
                    ax_y.plot([xi], [zi], 'yo')
                    ax_z.plot([xi], [yi], 'yo')
                elif im_labels[i] == 1: #Shower
                    ax_x.plot([yi], [zi], 'go')
                    ax_y.plot([xi], [zi], 'go')
                    ax_z.plot([xi], [yi], 'go')
            ax_x.set_xlim(0, cfg.IMAGE_SIZE)
            ax_x.set_ylim(0, cfg.IMAGE_SIZE)
            ax_y.set_xlim(0, cfg.IMAGE_SIZE)
            ax_y.set_ylim(0, cfg.IMAGE_SIZE)
            ax_z.set_xlim(0, cfg.IMAGE_SIZE)
            ax_z.set_ylim(0, cfg.IMAGE_SIZE)
        plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_projections_%d.png' % index), dpi=200)
        plt.close(fig3)"""

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

def closest_gt_distance(im_proposals, gt_pixels):
    gt_pixels = gt_pixels[:, :-1]
    distances = []
    for proposal in im_proposals:
        distances.append(np.min(np.sqrt(np.sum(np.power([proposal] - gt_pixels, 2), axis=1))))
    #print(im_proposals, gt_pixels, distances)
    return distances

def distances_plot(cfg, distances):
    plt.gcf().clear()
    bins = np.linspace(0, 5, 20)
    plt.hist(distances, 100)
    plt.xlabel("distance to closest ground truth pixel")
    plt.ylabel("#\"chosen\" pixels")
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, 'distances.png'))
    plt.gcf().clear()
    plt.hist(distances, bins)
    plt.xlabel("distance to closest ground truth pixel")
    plt.ylabel("#\"chosen\" pixels")
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, 'distances2.png'))
    plt.gcf().clear()
    plt.hist(distances, 10)
    plt.xlabel("distance to closest ground truth pixel")
    plt.ylabel("#\"chosen\" pixels")
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, 'distances3.png'))
    plt.gcf().clear()
    bins2 = np.linspace(0, 10, 100)
    plt.hist(distances, bins2)
    plt.xlabel("distance to closest ground truth pixel")
    plt.ylabel("#\"chosen\" pixels")
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, 'distances4.png'))

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
    im_proposals, im_labels, im_scores, distances = [], [], [], []
    with tf.Session() as sess:
        saver.restore(sess, cfg.WEIGHTS_FILE)
        for i in range(10):
            blob = data.forward()
            summary, results = net.test_image(sess, blob)
            if cfg.NET == 'ppn':
                im_labels.extend(results['im_labels'])
                im_scores.extend(results['im_scores'])
                im_proposals_filtered = display(blob, cfg, index=i, dim1=net.dim1, dim2=net.dim2, **results)
                distances.extend(closest_gt_distance(im_proposals_filtered, blob['gt_pixels']))
            else:
                print(blob, results)
    statistics(cfg, im_proposals, im_labels, im_scores)
    distances_plot(cfg, distances)


if __name__ == '__main__':
    inference(cfg)
