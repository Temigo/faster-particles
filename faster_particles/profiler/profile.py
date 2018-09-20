# *-* encoding: utf-8 *-*
# Demo for PPN
# Usage: python demo_ppn.py model.ckpt
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import glob
import time

from faster_particles.display_utils import display, display_uresnet, \
                                            display_ppn_uresnet
from faster_particles.ppn import PPN
from faster_particles.base_net.uresnet import UResNet
from faster_particles.base_net import basenets
from faster_particles.metrics import PPNMetrics, UResNetMetrics
from faster_particles.data import ToydataGenerator, LarcvGenerator, \
                                HDF5Generator, CSVGenerator
from faster_particles.cropping import cropping_algorithms
from faster_particles.demo_ppn import get_data, load_weights


def test_cropping(cfg):
    if not os.path.isdir(cfg.DISPLAY_DIR):
        os.makedirs(cfg.DISPLAY_DIR)

    _, data = get_data(cfg)
    crop_algorithm = cropping_algorithms[cfg.CROP_ALGO](cfg)
    duration1, duration2 = 0, 0
    for i in range(cfg.MAX_STEPS):
        blob = data.forward()
        print(np.count_nonzero(blob['data']))
        start = time.time()
        patch_centers, patch_sizes = crop_algorithm.crop(blob['voxels'])
        end = time.time()
        duration1 += end - start
        start = time.time()
        batch_blobs, patch_centers, patch_sizes = crop_algorithm.extract(patch_centers, patch_sizes, blob)
        end = time.time()
        duration2 += end - start
        print("Cropped into %d images" % len(patch_centers))
    duration1 /= cfg.MAX_STEPS
    duration2 /= cfg.MAX_STEPS
    print("Average duration = %f + %f" % (duration1, duration2))


def main(cfg):
    if not os.path.isdir(cfg.DISPLAY_DIR):
        os.makedirs(cfg.DISPLAY_DIR)

    _, data = get_data(cfg)

    net = PPN(cfg=cfg, base_net=basenets[cfg.BASE_NET])

    net.init_placeholders()
    net.create_architecture(is_training=True)
    duration = 0

    if cfg.PROFILE:
        print('WARNING PROFILING ENABLED')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # print(getargspec(self.sess.run))
        run_metadata = tf.RunMetadata()
        old_run = tf.Session.run
        new_run = lambda self, fetches, feed_dict=None: old_run(self, fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        tf.Session.run = new_run

    crop_algorithm = cropping_algorithms[cfg.CROP_ALGO](cfg)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(cfg, sess)
        real_step = 0
        for i in range(cfg.MAX_STEPS):
            print("%d/%d" % (i, cfg.MAX_STEPS))
            blob = data.forward()
            # Cropping pre-processing
            patch_centers, patch_sizes = None, None
            if cfg.ENABLE_CROP:
                batch_blobs, patch_centers, patch_sizes = crop_algorithm.process(blob)
            else:
                batch_blobs = [blob]

            for j, blob in enumerate(batch_blobs):
                real_step += 1
                feed_dict = {
                    net.image_placeholder: blob['data'],
                    net.gt_pixels_placeholder: blob['gt_pixels']
                    }
                print(j)
                start = time.time()
                # summary, results = net.test_image(sess, blob)
                # _ = sess.run([net.last_layer], feed_dict=feed_dict)
                # _ = sess.run([net._predictions['rois']], feed_dict=feed_dict)
                # _ = sess.run([net.rpn_pooling], feed_dict=feed_dict)
                # _ = sess.run([net._predictions['ppn2_proposals']], feed_dict=feed_dict)
                # _ = sess.run([net.before_nms], feed_dict=feed_dict)
                # _ = sess.run([net.after_nms], feed_dict=feed_dict)
                _ = sess.run([net._predictions['im_proposals']], feed_dict=feed_dict)
                # _ = sess.run([net.train_op], feed_dict=feed_dict)
                end = time.time()
                duration += end - start

    if cfg.PROFILE:
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(cfg.PROFILE_NAME, 'w') as f:
            f.write(ctf)
            print("Wrote timeline to %s" % cfg.PROFILE_NAME)

        # # Print to stdout an analysis of the memory usage and the timing information
        # # broken down by python codes.
        # ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
        # opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
        #     ).with_node_names(show_name_regexes=['*']).build()
        #
        # tf.profiler.profile(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     cmd='code',
        #     options=opts)
        #
        # # Print to stdout an analysis of the memory usage and the timing information
        # # broken down by operation types.
        # tf.profiler.profile(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     cmd='op',
        #     options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    duration /= cfg.MAX_STEPS
    print("Average duration of inference = %f ms" % duration)


if __name__ == '__main__':
    class MyCfg(object):
        # Cropping algorithms
        SLICE_SIZE = 64
        CORE_SIZE = 32
        ENABLE_CROP = False
        CROP_ALGO = "proba"
        MAX_PATCHES = 500  # for proba algo
        MIN_OVERLAP = 2  # for proba algo

        # General settings
        OUTPUT_DIR = "output"
        LOG_DIR = "log"
        DISPLAY_DIR = "display"
        MAX_STEPS = 100
        LEARNING_RATE = 0.001
        PROFILE = False
        PROFILE_TIMELINE = 'timeline.json'

        # PPN
        R = 20
        # Feature map indexes:
        # 0 is stride 0 (original image size).
        # See uresnet.py to get the spatial depth.
        # TODO Rename PPN1_INDEX (corresponds to PPN2) and PPN2_INDEX (~ PPN1)
        PPN1_INDEX = 2  # Index of the intermediate feature map in PPN
        PPN2_INDEX = -1  # Index of the final feature map in PPN
        PPN1_SCORE_THRESHOLD = 0.6
        PPN2_DISTANCE_THRESHOLD = 5
        LAMBDA_PPN = 0.5
        LAMBDA_PPN1 = 0.5
        LAMBDA_PPN2 = 0.5
        WEIGHTS_FILE = None  # Path to pretrained checkpoint
        WEIGHTS_FILE_BASE = None
        WEIGHTS_FILE_PPN = None
        WEIGHTS_FILE_SMALL = None
        FREEZE = False  # Whether to freeze the weights of base net
        NET = 'ppn'
        BASE_NET = 'vgg'
        WEIGHT_LOSS = False
        MIN_SCORE = 0.0
        POSTPROCESSING = 'nms'  # Postprocessing: use either NMS or DBSCAN

        # UResNet
        URESNET_WEIGHTING = False  # Use pixel-weighting scheme in UResNet
        URESNET_ADD = False  # Whether to use add or concat in UResNet
        NUM_CLASSES = 3  # For base network only
        BASE_NUM_OUTPUTS = 16  # For UResNet
        NUM_STRIDES = 5  # spatial depth for UResNet

        IMAGE_SIZE = 192
        BASE_NET = 'uresnet'
        NET = 'ppn'
        ENABLE_CROP = False
        SLICE_SIZE = 64
        MAX_STEPS = 10
        CROP_ALGO = 'proba'
        DISPLAY_DIR = 'display/profile'
        OUTPUT_DIR = 'output/profile'
        LOG_DIR = 'log/profile'
        WEIGHTS_FILE_BASE = '/data/train_slicing9/model-40000.ckpt'
        DATA = '/data/dlprod_ppn_v08_p02_filtered/train_p02.root'
        TEST_DATA = '/data/dlprod_ppn_v08_p02_filtered/test_p02.root'
        DATA_TYPE = 'larcv'
        GPU = '0'
        TOYDATA = False
        DATA_3D = True
        SEED = 123
        NUM_CLASSES = 3
        LEARNING_RATE = 0.001
        BASE_NUM_OUTPUTS = 16
        WEIGHTS_FILE_PPN = None
        URESNET_WEIGHTING = False
        URESNET_ADD = False
        PPN2_INDEX = 5
        PPN1_INDEX = 3
        NUM_STRIDES = 5
        PROFILE = False
        PROFILE_NAME = 'timeline_ppn_3_5_memory2.json'
        NEXT_INDEX = 0
        BATCH_SIZE = 1
        R = 20

    cfg = MyCfg()
    main(cfg)
    # test_cropping(cfg)
