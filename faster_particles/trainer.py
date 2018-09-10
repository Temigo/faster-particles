# *-* encoding: utf-8 *-*
# Trainer class for PPN, base network and small UResNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
import numpy as np

from faster_particles.demo_ppn import load_weights
from faster_particles.display_utils import extract_voxels, draw_slicing
from faster_particles.cropping import cropping_algorithms


class Trainer(object):
    def __init__(self, net, train_toydata, test_toydata,
                 cfg, display_util=None):
        self.train_toydata = train_toydata
        self.test_toydata = test_toydata
        self.net = net
        self.display = display_util
        self.cfg = cfg
        self.dim = 3 if self.cfg.DATA_3D else 2

        self.logdir = cfg.LOG_DIR
        self.displaydir = cfg.DISPLAY_DIR
        self.outputdir = cfg.OUTPUT_DIR
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.isdir(self.displaydir):
            os.makedirs(self.displaydir)
        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)

    def process_blob(self, i, blob, real_step, saver, is_testing,
                     summary_writer_train, summary_writer_test):
        real_step += 1
        is_drawing = real_step % 1000 == 0

        if real_step % 100 == 0:
            print("(Real) Step %d" % real_step)

        if self.cfg.NET == 'small_uresnet':
            blob['data'] = np.reshape(
                blob['crops'],
                (-1,) + (self.cfg.CROP_SIZE,) * self.dim + (1,))
            blob['labels'] = np.reshape(
                blob['crops_labels'],
                (-1,) + (self.cfg.CROP_SIZE,) * self.dim)
            if is_testing:
                summary, result = self.test_net.test_image(self.sess, blob)
                summary_writer_test.add_summary(summary, real_step)
            else:
                summary, result = self.train_net.train_step(self.sess, blob)
                summary_writer_train.add_summary(summary, real_step)
            for i in range(len(blob['crops'])):
                blob_i = {
                    'data': np.reshape(
                        blob['crops'][i],
                        (1,) + (self.cfg.CROP_SIZE,) * self.dim + (1,)
                        ),
                    'labels': np.reshape(
                        blob['crops_labels'][i],
                        (1,) + (self.cfg.CROP_SIZE,) * self.dim
                        )
                    }
                if is_drawing and self.display is not None:
                    N = self.cfg.IMAGE_SIZE
                    self.cfg.IMAGE_SIZE = self.cfg.CROP_SIZE
                    self.display(blob_i,
                                 self.cfg,
                                 index=real_step,
                                 name='display_train',
                                 directory=os.path.join(
                                     self.cfg.DISPLAY_DIR,
                                     'train'),
                                 vmin=0,
                                 vmax=1,
                                 predictions=np.reshape(
                                     result['predictions'][i],
                                     (1,) + (self.cfg.CROP_SIZE,) * self.dim
                                     )
                                 )
                    self.cfg.IMAGE_SIZE = N
        else:
            if is_testing:
                summary, result = self.test_net.test_image(self.sess, blob)
                summary_writer_test.add_summary(summary, real_step)
            else:
                summary, result = self.train_net.train_step(self.sess, blob)
                summary_writer_train.add_summary(summary, real_step)

            if is_drawing and self.display is not None:
                print('Drawing...')
                if self.cfg.NET == 'ppn':
                    result['dim1'] = self.train_net.dim1
                    result['dim2'] = self.train_net.dim2
                if self.cfg.ENABLE_CROP:
                    N = self.cfg.IMAGE_SIZE
                    self.cfg.IMAGE_SIZE = self.cfg.SLICE_SIZE
                self.display(blob,
                             self.cfg,
                             index=real_step,
                             name='display_train',
                             directory=os.path.join(self.cfg.DISPLAY_DIR,
                                                    'train'),
                             **result)
                if self.cfg.ENABLE_CROP:
                    self.cfg.IMAGE_SIZE = N
                print("Done.")

        if real_step % 1000 == 0:
            save_path = saver.save(self.sess,
                                   os.path.join(self.outputdir,
                                                "model-%d.ckpt" % real_step))
            print("Wrote %s" % save_path)
            print("Memory usage: ", self.sess.run(tf.contrib.memory_stats.MaxBytesInUse()))

        return real_step, result

    def train(self, net_args, scope="ppn"):
        print("Creating net architecture...")
        net_args['cfg'] = self.cfg
        self.train_net = self.net(**net_args)
        self.test_net = self.net(**net_args)
        self.test_net.restore_placeholder(self.train_net.init_placeholders())
        self.train_net.create_architecture(is_training=True,
                                           reuse=False,
                                           scope=scope)
        self.test_net.create_architecture(is_training=False,
                                          reuse=True,
                                          scope=scope)
        print("Done.")

        # with tf.Session() as sess:
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        load_weights(self.cfg, self.sess)
        summary_writer_train = tf.summary.FileWriter(
            os.path.join(self.logdir, 'train'), self.sess.graph)
        summary_writer_test = tf.summary.FileWriter(
            os.path.join(self.logdir, 'test'), self.sess.graph)

        # Global saver
        # saver = None
        # if self.cfg.FREEZE: # Save only PPN weights (excluding base network)
        #     variables_to_restore = [v for v in tf.global_variables() if "ppn" in v.name]
        #     saver = tf.train.Saver(variables_to_restore)
        # else: # Save everything (including base network)
        saver = tf.train.Saver()

        step = 0
        crop_algorithm = cropping_algorithms[self.cfg.CROP_ALGO](self.cfg)

        self.batch_size = self.cfg.BATCH_SIZE
        self.cfg.BATCH_SIZE = 1

        print("Start training...")
        real_step = 0
        for step in range(self.cfg.MAX_STEPS):
            sys.stdout.flush()
            is_testing = step % 10 == 5
            is_drawing = step % 200 == 0
            if is_testing:
                blob = self.test_toydata.forward()
            else:
                blob = self.train_toydata.forward()
            if step % 10 == 0:
                print("Iteration %d/%d" % (step, self.cfg.MAX_STEPS))

            # Cropping pre-processing
            patch_centers, patch_sizes = None, None
            if self.cfg.ENABLE_CROP:
                batch_blobs, patch_centers, patch_sizes = crop_algorithm.process(blob)
                if is_drawing:
                    draw_slicing(blob, self.cfg, patch_centers, patch_sizes,
                                 index=step, name='slices',
                                 directory=os.path.join(self.cfg.DISPLAY_DIR, 'cropping'))
            else:
                batch_blobs = [blob]

            i = 0
            batch_results = []
            while i < len(batch_blobs):
                blobs = batch_blobs[i:min(i+self.batch_size, len(batch_blobs))]
                blob = {}
                for key in blobs[0]:
                    blob[key] = np.concatenate([b[key] for b in blobs])

                i += len(blobs)
                real_step, result = self.process_blob(i, blob, real_step,
                                                      saver, is_testing,
                                                      summary_writer_train,
                                                      summary_writer_test)
                """for j in range(len(blobs)):
                    r = {}
                    for key in result:
                        r[key] = result[key][j]
                    batch_results.append(r)"""

            # Reconcile slices result together
            # using batch_results, batch_blobs, patch_centers and patch_sizes
            """final_results = {}
            # UResNet predictions
            if 'predictions' and 'scores' in batch_results[0]:
                final_voxels = []  # Shape N_voxels x dim
                final_scores = []  # Shape N_voxels x num_classes
                final_counts = []  # Shape N_voxels x 1
                for i, blob in enumerate(batch_blobs):
                    result = batch_results[i]
                    v = extract_voxels(result['predictions'])  # Shape N_voxels x dim
                    scores = result['softmax'][v]  # Shape N_voxels x num_classes
                    # indices are  indices of the *first* occurrences of the unique values
                    # hence for doublons they are indices in final_voxels
                    final_voxels, indices, counts = np.unique(np.concatenate([final_voxels, v], axis=0), axis=0, return_index=True, return_counts=True)
                    final_scores = np.concatenate([final_scores, scores], axis=0)[indices]
                    final_counts = np.concatenate([final_counts, np.ones_like(v)], axis=0)[indices] + counts - 1
                final_scores = final_scores / final_counts  # Compute average
                final_predictions = np.argmax(final_scores, axis=1)
                final_results['predictions'] = final_predictions
                final_results['scores'] = final_scores[final_predictions]
                final_results['softmax'] = final_scores"""

        summary_writer_train.close()
        summary_writer_test.close()
        print("Done.")
