# *-* encoding: utf-8 *-*
# Trainer class for PPN, base network and small UResNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.client import timeline
import os
import sys
import numpy as np
from functools import partial
from inspect import getargspec

from faster_particles.demo_ppn import load_weights
from faster_particles.display_utils import draw_slicing
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
                     summary_writer_train, summary_writer_test, run_metadata):
        """
        Runs 1 training iteration on blob.
        """
        real_step += 1
        is_drawing = real_step > 0 and real_step % 1000 == 0

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
            # print(blob['entries'])
            # print(np.sum(blob['weight']), np.amin(blob['weight']), np.amax(blob['weight']))
            # print(np.unique(blob['weight'], return_counts=True))
            if is_testing:
                summary, result = self.test_net.test_image(self.sess, blob)
                summary_writer_test.add_summary(summary, real_step)
                if self.cfg.PROFILE:
                    summary_writer_test.add_run_metadata(run_metadata, "step_%d" % real_step, real_step)
            else:
                summary, result = self.train_net.train_step(self.sess, blob)
                summary_writer_train.add_summary(summary, real_step)
                if self.cfg.PROFILE:
                    summary_writer_train.add_run_metadata(run_metadata, "step_%d" % real_step, real_step)

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
        """
        Main training function.
        """
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
        if self.cfg.NET in ['ppn', 'ppn_ext', 'full']:
            self.cfg.dim1 = self.train_net.dim1
            self.cfg.dim2 = self.train_net.dim2
        print("Done.")

        run_metadata = None
        if self.cfg.PROFILE:
            print('WARNING PROFILING ENABLED')
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # print(getargspec(self.sess.run))
            run_metadata = tf.RunMetadata()
            builder = tf.profiler.ProfileOptionBuilder
            opts = builder(builder.time_and_memory()).order_by('micros').build()
            pctx = tf.contrib.tfprof.ProfileContext('/data/profiler',
                                      trace_steps=[],
                                      dump_steps=[])
            old_run = tf.Session.run
            def new_run(self, fetches, feed_dict=None):
                pctx.trace_next_step()
                pctx.dump_next_step()
                results = old_run(self, fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                # pctx.profiler.profile_operations(options=opts)
                return results

            # new_run = lambda self, fetches, feed_dict=None: old_run(self, fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            # self.sess.run = partial(self.sess.run,
            #                         options=run_options,
            #                         run_metadata=run_metadata)
            # self.sess.run_old = self.sess.run
            tf.Session.run = new_run

        # with tf.Session() as sess:
        self.sess = tf.Session()

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
        saver = tf.train.Saver(max_to_keep=(0 if self.cfg.DETAIL_LOG else 5),
                               keep_checkpoint_every_n_hours=(0.5 if self.cfg.DETAIL_LOG else 10000.0))

        step = 0
        crop_algorithm = cropping_algorithms[self.cfg.CROP_ALGO](self.cfg)

        self.batch_size = self.cfg.BATCH_SIZE
        self.cfg.BATCH_SIZE = 1

        print("Start training...")
        real_step = 0
        for step in range(self.cfg.MAX_STEPS):
            sys.stdout.flush()
            is_testing = step % 10 == 5
            is_drawing = step > 0 and step % 200 == 0
            if is_testing:
                blob = self.test_toydata.forward()
            else:
                blob = self.train_toydata.forward()
            if step % 10 == 0:
                print("Iteration %d/%d" % (step, self.cfg.MAX_STEPS))

            # Cropping pre-processing
            # print(blob['entries'])
            patch_centers, patch_sizes = None, None
            if self.cfg.ENABLE_CROP:
                batch_blobs, patch_centers, patch_sizes = crop_algorithm.process(blob)
                if is_drawing:
                    draw_slicing(blob, self.cfg, patch_centers, patch_sizes,
                                 index=step, name='slices',
                                 directory=os.path.join(self.cfg.DISPLAY_DIR,
                                                        'cropping'))
                    print("Cropping %d patches..." % len(patch_centers))
                    print("Overlap: ",
                          crop_algorithm.compute_overlap(blob['voxels'],
                                               patch_centers,
                                               sizes=patch_sizes[:, np.newaxis]))
            else:
                batch_blobs = [blob]

            i = 0
            batch_results = []
            while i+self.batch_size <= len(batch_blobs):
                blobs = batch_blobs[i:i+self.batch_size]
                miniblob = {}
                for key in blobs[0]:
                    miniblob[key] = np.concatenate([b[key] for b in blobs])

                i += self.batch_size
                real_step, result = self.process_blob(i, miniblob, real_step,
                                                      saver, is_testing,
                                                      summary_writer_train,
                                                      summary_writer_test,
                                                      run_metadata)
                # Temporary - check whether there are empty slices
                x = np.sum(miniblob['data'], axis=(1, 2, 3, 4))
                if not np.all(x > 0.0):
                    print("STOP", x)
                # Keep results for synthesis later
                for j in range(len(blobs)):
                    r = {}
                    for key in result:
                        if key in ['im_proposals', 'im_scores', 'im_labels', 'rois']:
                            r[key] = result[key]
                        elif key in ['dim1', 'dim2']:
                            pass
                        else:
                            r[key] = result[key][j]
                    batch_results.append(r)

            if self.cfg.ENABLE_CROP:
                final_results = crop_algorithm.reconcile(batch_results,
                                                         patch_centers,
                                                         patch_sizes)

                if is_drawing:
                    self.display(blob,
                                 self.cfg,
                                 index=step,
                                 name='display_train_final',
                                 directory=os.path.join(self.cfg.DISPLAY_DIR,
                                                        'train'),
                                 **final_results)

        summary_writer_train.close()
        summary_writer_test.close()
        print("Done.")

        if self.cfg.PROFILE:
            pctx.profiler.profile_operations(options=opts)

            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(self.cfg.PROFILE_TIMELINE, 'w') as f:
                f.write(ctf)
                print("Wrote timeline to %s" % self.cfg.PROFILE_TIMELINE)

            # Print to stdout an analysis of the memory usage and the timing information
            # broken down by python codes.
            ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
            # opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
            #     ).with_node_names(show_name_regexes=['.*trainer.py.*']).build()
            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            tf.profiler.profile(
                tf.get_default_graph(),
                run_meta=run_metadata,
                cmd='code',
                options=opts)

            # Print to stdout an analysis of the memory usage and the timing information
            # broken down by operation types.
            tf.profiler.profile(
                tf.get_default_graph(),
                run_meta=run_metadata,
                cmd='op',
                options=tf.profiler.ProfileOptionBuilder.time_and_memory())
