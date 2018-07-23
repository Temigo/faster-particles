# *-* encoding: utf-8 *-*
# Demo for PPN
# Usage: python demo_ppn.py model.ckpt
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys, os, subprocess, glob

from faster_particles.display_utils import display, display_uresnet, display_ppn_uresnet
from faster_particles.ppn import PPN
from faster_particles.base_net import basenets
from faster_particles import ToydataGenerator
from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator
from faster_particles.metrics import PPNMetrics

CLASSES = ('__background__', 'track_edge', 'shower_start', 'track_and_shower')

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

def inference_simple(cfg, blobs, net, num_test=10):
    net.init_placeholders()
    net.create_architecture(is_training=False)
    inference = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(cfg, sess)
        for i in range(num_test):
            summary, results = net.test_image(sess, blobs[i])
            inference.append(results)
    return inference

def inference_full(cfg):
    #if cfg.WEIGHTS_FILE_BASE is None or cfg.WEIGHTS_FILE_PPN is None:
    #    raise Exception("Need both weights files for full inference.")

    num_test = cfg.MAX_STEPS
    inference_base, inference_ppn, blobs = [], [], []
    weights_file_ppn = cfg.WEIGHTS_FILE_PPN
    print("Retrieving data...")
    data = get_data(cfg)
    for i in range(num_test):
        blobs.append(data.forward())
    print("Done.")

    # First base
    print("Base network...")
    cfg.WEIGHTS_FILE_PPN = None
    net_base = basenets[cfg.BASE_NET](cfg=cfg)
    inference_base = inference_simple(cfg, blobs, net_base, num_test=num_test)
    print("Done.")
    print(inference_base)
    tf.reset_default_graph()
    print("PPN network...")
    cfg.WEIGHTS_FILE_PPN = weights_file_ppn
    net_ppn = PPN(cfg=cfg, base_net=basenets[cfg.BASE_NET])
    inference_ppn = inference_simple(cfg, blobs, net_ppn, num_test=num_test)
    print("Done.")
    print(inference_ppn)

    # Display
    print("Saving displays...")
    metrics = PPNMetrics(cfg, dim1=net_ppn.dim1, dim2=net_ppn.dim2)
    for i in range(num_test):
        #results = {**inference_base[i], **inference_ppn[i]}
        results = inference_base[i].copy()
        results.update(inference_ppn[i])
        im_proposals_filtered = display_ppn_uresnet(
            blobs[i],
            cfg,
            index=i,
            directory=os.path.join(cfg.DISPLAY_DIR, 'demo_full'),
            **results
        )
        metrics.add(blobs[i], results, im_proposals_filtered)
    metrics.plot()
    print("Done.")
    # Clustering: k-means? DBSCAN?

def clustering(inference_base, inference_ppn, blobs):
    # Rough clustering
    num_test = len(inference_base)
    eps=20
    for i in range(num_test):
        db = DBSCAN(eps=eps, min_samples=10).fit_predict(blobs[i]['voxels'])
        print(db)

    # Eliminate clusters unrelated to PPN points

    # Fine clustering

def inference(cfg):
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

    if is_ppn:
        metrics = PPNMetrics(cfg, dim1=net.dim1, dim2=net.dim2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(cfg, sess)

        for i in range(cfg.MAX_STEPS):
            print("%d/%d" % (i, cfg.MAX_STEPS))
            blob = data.forward()
            summary, results = net.test_image(sess, blob)
            if is_ppn:
                im_proposals_filtered = display(
                    blob,
                    cfg,
                    index=i,
                    dim1=net.dim1,
                    dim2=net.dim2,
                    directory=os.path.join(cfg.DISPLAY_DIR, 'demo'),
                    **results
                )
                metrics.add(blob, results, im_proposals_filtered)
            else:
                if cfg.BASE_NET == 'uresnet':
                    display_uresnet(blob, cfg, index=i, **results)
                else:
                    print(blob, results)

    if is_ppn:
        metrics.plot()

if __name__ == '__main__':
    #inference(cfg)
    inference_full(cfg)
