from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import sys, os, subprocess, glob, time
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy

from faster_particles.display_utils import display, display_uresnet, display_ppn_uresnet
from faster_particles.ppn import PPN
from faster_particles.base_net.uresnet import UResNet
from faster_particles.base_net import basenets
from faster_particles import ToydataGenerator
from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator
from faster_particles.metrics import PPNMetrics


def load_weights(cfg, sess):
    print("Restoring checkpoint file...")
    scopes = []
    if cfg.WEIGHTS_FILE_PPN is not None:
        scopes.append((lambda x: True, cfg.WEIGHTS_FILE_PPN))
    # Restore variables for base net if given checkpoint file
    elif cfg.WEIGHTS_FILE_BASE is not None:
        if cfg.NET == 'ppn': # load only relevant layers of base network
		    scopes.append((lambda x: cfg.BASE_NET in x and "optimizer" not in x, cfg.WEIGHTS_FILE_BASE))
            #scopes.append((lambda x: cfg.BASE_NET in x, cfg.WEIGHTS_FILE_BASE))
        else: # load for full base network
            scopes.append((lambda x: cfg.BASE_NET in x, cfg.WEIGHTS_FILE_BASE))


    for scope, weights_file in scopes:
        print('Restoring %s...' % weights_file)
        variables_to_restore = [v for v in tf.global_variables() if scope(v.name)]
        print("- ignoring %d/%d variables" % (len(tf.global_variables()) - len(variables_to_restore), len(tf.global_variables())))
        if len(variables_to_restore) > 0:
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, weights_file)
        else:
            print("WARNING No variable was restored from weights file %s." % weights_file)
    print("Done.")
    
# Returns a list of files as a string *without spaces*
# e.g. "["file1.root","file2.root"]"
def get_filelist(ls_command):
    #filelist = subprocess.Popen(["ls %s" % ls_command], shell=True, stdout=subprocess.PIPE).stdout
    filelist = glob.glob(ls_command)
    print(str(filelist))
    return str(filelist).replace('\'', '\"').replace(" ", "")


def get_data(cfg):
    if cfg.TOYDATA:
        if cfg.NET == 'ppn':
            data = ToydataGenerator(cfg)
        else:
            data = ToydataGenerator(cfg, classification=True)
    else:
        filelist = get_filelist(cfg.DATA)
        data = LarcvGenerator(cfg, ioname="inference", filelist=filelist)
    return data

def inference_k(cfg):
    data = get_data(cfg)
    is_ppn = cfg.NET == 'ppn'
    if is_ppn:
        net = PPN(cfg=cfg, base_net=basenets[cfg.BASE_NET])
        if cfg.WEIGHTS_FILE_PPN is None:
            pass
            #raise Exception("Need a checkpoint file for PPN at least")
    elif cfg.NET == 'base':
        net = basenets[cfg.BASE_NET](cfg=cfg)
        if cfg.WEIGHTS_FILE_PPN is None and cfg.WEIGHTS_FILE_BASE is None:
            raise Exception("Need a checkpoint file")

    net.init_placeholders()
    net.create_architecture(is_training=False)
    ppn2_distances_to_closest_gt = []
    ppn2_distances_to_closest_pred = []
    duration = 0
    if is_ppn:
        metrics = PPNMetrics(cfg, dim1=net.dim1, dim2=net.dim2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(cfg, sess)

        for i in range(cfg.MAX_STEPS):
            print("%d/%d" % (i, cfg.MAX_STEPS))
            blob = data.forward()
            start = time.time()
            summary, results = net.test_image(sess, blob)
            end = time.time()
            duration += end - start
            if is_ppn:
                display(
                    blob,
                    cfg,
                    index=i,
                    dim1=net.dim1,
                    dim2=net.dim2,
                    directory=os.path.join(cfg.DISPLAY_DIR, 'demo'),
                    **results
                )
                gt_pixels = blob['gt_pixels'][:, :-1]
                im_proposals = results['im_proposals']
                distances_ppn2 = scipy.spatial.distance.cdist(im_proposals, gt_pixels)
                ppn2_distances_to_closest_gt.extend(np.amin(distances_ppn2, axis=0))
                ppn2_distances_to_closest_pred.extend(np.amin(distances_ppn2, axis=1))
            else:
                if cfg.BASE_NET == 'uresnet':
                    display_uresnet(blob, cfg, index=i, **results)
                else:
                    print(blob, results)
    duration /= cfg.MAX_STEPS
    print("Average duration of inference = %f ms" % duration)
    
    dirpath = os.path.join(cfg.DISPLAY_DIR, 'metrics')
    if is_ppn:
        plot_distances_to_closest_gt(ppn2_distances_to_closest_gt, dirpath)
        plot_distances_to_closest_pred(ppn2_distances_to_closest_pred, dirpath)

def make_plot(data, dirpath, bins=None, xlabel="", ylabel="", filename=""):
    data = np.array(data)
    if bins is None:
        d = np.diff(np.unique(data)).min()
        left_of_first_bin = data.min() - float(d)/2
        right_of_last_bin = data.max() + float(d)/2
        bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

    plt.hist(data, bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(dirpath, filename))
    plt.gcf().clear()
            
def plot_distances_to_closest_gt(data, dirpath):
    bins = np.linspace(0, 100, 100)
    make_plot(
        data,
        dirpath,
        bins=50,
        xlabel="distance to nearest ground truth pixel",
        ylabel="#proposed pixels",
        filename='ppn2_distance_to_closest_gt.png'
    )
    make_plot(
        data,
        dirpath,
        bins=np.linspace(0, 5, 100),
        xlabel="distance to nearest ground truth pixel",
        ylabel="#proposed pixels",
        filename='ppn2_distance_to_closest_gt_zoom.png'
    )
        
def plot_distances_to_closest_pred(data, dirpath):
    bins = np.linspace(0, 100, 100)
    make_plot(
        data,
        dirpath,
        bins=50,
        xlabel="distance to nearest proposed pixel",
        ylabel="#ground truth pixels",
        filename='ppn2_distance_to_closest_pred.png'
    )
    make_plot(
        data,
        dirpath,
        bins=np.linspace(0, 5, 100),
        xlabel="distance to nearest proposed pixel",
        ylabel="#ground truth pixels",
        filename='ppn2_distance_to_closest_pred_zoom.png'
    )