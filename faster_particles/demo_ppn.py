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

def filter_points(im_proposals, im_scores, eps):
    db = DBSCAN(eps=eps, min_samples=1).fit_predict(im_proposals)
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

def distance_to_nearest_neighbour(cfg, im_proposals):
    # FIXME vectorize loop
    distances = []
    for point in im_proposals:
        distances.append(np.partition(np.sum(np.power(point - im_proposals, 2), axis=1), 2)[1])
    bins = np.linspace(0, 100, 100)
    plt.hist(distances, bins)
    plt.xlabel("distance to nearest neighbour")
    plt.ylabel("#proposals")
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, 'distance_to_nearest_neighbour.png'))
    return distances

def display_original_image(blob, cfg, ax, vmin=0, vmax=400, cmap='jet'):
    # Display original image
    if cfg.DATA_3D:
        for i in range(len(blob['voxels'])):
            voxel = blob['voxels'][i]
            if 'voxels_value' in blob:

                if blob['voxels_value'][i] == 1: # track
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='red', alpha=0.3, linewidths=0.0, edgecolors='black')
                elif blob['voxels_value'][i] == 2: # shower
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='blue', alpha=0.3, linewidths=0.0, edgecolors='black')
                else:
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='black', alpha=0.3, linewidths=0.0, edgecolors='black')
            else:
                draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='blue', alpha=0.3, linewidths=0.1, edgecolors='black')
    else:
        ax.imshow(blob['data'][0,...,0], cmap=cmap, interpolation='none', origin='lower', vmin=vmin, vmax=vmax)

def set_image_limits(cfg, ax):
    ax.set_xlim(0, cfg.IMAGE_SIZE)
    ax.set_ylim(0, cfg.IMAGE_SIZE)
    if cfg.DATA_3D:
        ax.set_zlim(0, cfg.IMAGE_SIZE)

def extract_voxels(data):
    indices = np.where(data > 0)
    return np.stack(indices).T, data[indices]

def display_uresnet(blob, cfg, index=0, predictions=None, name='display'):
    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'
        blob['voxels'], blob['voxels_value'] = extract_voxels(blob['data'][0,...,0])

    if predictions is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', **kwargs)

        display_original_image(blob, cfg, ax, vmax=10)

        set_image_limits(cfg, ax)

        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_original_%d.png' % index))
        plt.close(fig)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
        blob_label = {}
        blob_label['data'] = blob['labels'][0,...]
        blob_label['voxels'], blob_label['voxels_value'] = extract_voxels(blob['labels'][0,...])
        display_original_image(blob_label, cfg, ax2, vmax=3.1, cmap='tab10')

        set_image_limits(cfg, ax2)
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_labels_%d.png' % index))
        plt.close(fig2)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, aspect='equal', **kwargs)
        blob_pred = {}
        blob_pred['data'] = predictions[0,...]
        blob_pred['voxels'], blob_pred['voxels_value'] = extract_voxels(predictions[0,...])
        print(blob_pred['voxels'], blob_pred['voxels_value'])
        display_original_image(blob_pred, cfg, ax3, vmax=3.1)

        set_image_limits(cfg, ax3)
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_predictions_%d.png' % index))
        plt.close(fig3)

def display(blob, cfg, im_proposals=None, rois=None, im_labels=None, im_scores=None,
            index=0, dim1=8, dim2=4, name='display'):
    print(im_proposals)
    print(im_scores)
    print(im_labels)

    N = blob['data'].shape[1]
    N2 = int(N/dim1) # F3 size
    N3 = int(N2/dim2) # F5 size
    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', **kwargs)

    display_original_image(blob, cfg, ax)

    # display gt pixels
    if cfg.DATA_3D:
        for gt_pixel in blob['gt_pixels']:
            x, y, z = gt_pixel[2], gt_pixel[1], gt_pixel[0]
            draw_voxel(x, y, z, 1, ax, facecolors='red', alpha=1.0, linewidths=0.3, edgecolors='red')
    """else:
        for gt_pixel in blob['gt_pixels']:
            x, y = gt_pixel[1], gt_pixel[0]
            if gt_pixel[2] == 1:
                plt.plot([x], [y], 'ro')
            elif gt_pixel[2] == 2:
                plt.plot([x], [y], 'go')"""


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

    set_image_limits(cfg, ax)

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
            eps = 20.0 #9.0
            if cfg.DATA_3D:
                eps = 15.0 # FIXME
            im_proposals = filter_points(im_proposals, im_scores, eps)
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
    set_image_limits(cfg, ax2)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(cfg.DISPLAY_DIR, name + '_predictions_%d.png' % index))
    plt.close(fig2)
    return im_proposals

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

def inference_simple(cfg, net, weights_file, num_test=10):
    if cfg.TOYDATA:
        if cfg.NET == 'ppn':
            data = ToydataGenerator(cfg)
        else:
            data = ToydataGenerator(cfg, classification=True)
    else:
        filelist = get_filelist(cfg.DATA)
        data = LarcvGenerator(cfg, ioname="inference", filelist=filelist)

    net.init_placeholders()
    net.create_architecture(is_training=False)
    saver = tf.train.Saver()
    inference = []
    with tf.Session() as sess:
        saver.restore(sess, weights_file)
        for i in range(num_test):
            blob = data.forward()
            summary, results = net.test_image(sess, blob)
            inference.append(results)
    return inference

def inference_full(cfg):
    num_test = 10
    # First base
    net_base = basenets[cfg.BASE_NET](cfg=cfg)
    inference_base = inference_simple(cfg, net, cfg.WEIGHTS_BASE_FILE, num_test=num_test)
    # Then PPN
    net_ppn = PPN(cfg=cfg)
    inference_ppn = inference_simple(cfg, net, cfg.WEIGHTS_FILE, num_test=num_test)
    print(inference_base, inference_ppn)
    # Clustering: k-means? DBSCAN?

def inference(cfg):
    if cfg.TOYDATA:
        if cfg.NET == 'ppn':
            data = ToydataGenerator(cfg)
        else:
            data = ToydataGenerator(cfg, classification=True)
    else:
        filelist = get_filelist(cfg.DATA)
        data = LarcvGenerator(cfg, ioname="inference", filelist=filelist)

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
                im_proposals.extend(results['im_proposals'])
            else:
                if cfg.BASE_NET == 'uresnet':
                    display_uresnet(blob, cfg, index=i, **results)
                else:
                    print(blob, results)
    statistics(cfg, im_proposals, im_labels, im_scores)
    distances_plot(cfg, distances)
    distance_to_nearest_neighbour(cfg, im_proposals)

if __name__ == '__main__':
    inference(cfg)
