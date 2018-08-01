# *-* encoding: utf-8 *-*
# Trainer for both PPN and base network

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

from faster_particles.ppn import PPN
from faster_particles import ToydataGenerator
#from faster_particles import LarcvGenerator
from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator
from faster_particles.demo_ppn import get_filelist, load_weights
from faster_particles.display_utils import display, display_uresnet, display_ppn_uresnet
from faster_particles.base_net import basenets


class Trainer(object):
    def __init__(self, net, train_toydata, test_toydata, cfg, display_util=None):
        self.train_toydata = train_toydata
        self.test_toydata = test_toydata
        self.net = net
        self.logdir = cfg.LOG_DIR
        self.displaydir = cfg.DISPLAY_DIR
        self.outputdir = cfg.OUTPUT_DIR
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.isdir(self.displaydir):
            os.makedirs(self.displaydir)
        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)

        self.display = display_util
        self.cfg = cfg

    def train(self, net_args, scope="ppn"):
        print("Creating net architecture...")
        net_args['cfg'] = self.cfg
        self.train_net = self.net(**net_args)
        self.test_net = self.net(**net_args)
        self.test_net.restore_placeholder(self.train_net.init_placeholders())
        self.train_net.create_architecture(is_training=True, reuse=False, scope=scope)
        self.test_net.create_architecture(is_training=False, reuse=True, scope=scope)
        print("Done.")

        #with tf.Session() as sess:
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        load_weights(self.cfg, sess)
        summary_writer_train = tf.summary.FileWriter(os.path.join(self.logdir, 'train'), sess.graph)
        summary_writer_test = tf.summary.FileWriter(os.path.join(self.logdir, 'test'), sess.graph)


        # Global saver
        #saver = None
        #if self.cfg.FREEZE: # Save only PPN weights (excluding base network)
        #    variables_to_restore = [v for v in tf.global_variables() if "ppn" in v.name]
        #    saver = tf.train.Saver(variables_to_restore)
        #else: # Save everything (including base network)
        saver = tf.train.Saver()

        step = 0
        print("Start training...")
        while step < self.cfg.MAX_STEPS+1:
            step += 1
            is_testing = step%10 == 5
            is_drawing = step%1000 == 0
            if is_testing:
                blob = self.test_toydata.forward()
            else:
                blob = self.train_toydata.forward()

            if step%100 == 0:
                print("Step %d" % step)

            if self.cfg.NET == 'small_uresnet':
                blob['data'] = np.reshape(blob['crops'], (-1, self.cfg.CROP_SIZE, self.cfg.CROP_SIZE, 1))
                blob['labels'] = np.reshape(blob['crops_labels'], (-1, self.cfg.CROP_SIZE, self.cfg.CROP_SIZE))
                if is_testing:
                    summary, result = self.test_net.test_image(sess, blob)
                    summary_writer_test.add_summary(summary, step)
                else:
                    summary, result = self.train_net.train_step_with_summary(sess, blob)
                    summary_writer_train.add_summary(summary, step)
                for i in range(len(blob['crops'])):
                    blob_i = {'data': np.reshape(blob['crops'][i], (1, self.cfg.CROP_SIZE, self.cfg.CROP_SIZE, 1)), 'labels': np.reshape(blob['crops_labels'][i], (1, self.cfg.CROP_SIZE, self.cfg.CROP_SIZE))}
                    #print(result)
                    #print(blob_i['labels'])
                    if is_drawing and self.display is not None:
                        N = self.cfg.IMAGE_SIZE
                        self.cfg.IMAGE_SIZE = self.cfg.CROP_SIZE
                        self.display(blob_i, self.cfg, index=step, name='display_train', directory=os.path.join(self.cfg.DISPLAY_DIR, 'train'), vmin=0, vmax=1, predictions=np.reshape(result['predictions'][i], (1, self.cfg.CROP_SIZE, self.cfg.CROP_SIZE)))
                        self.cfg.IMAGE_SIZE = N
            else:
                if is_testing:
                    summary, result = self.test_net.test_image(sess, blob)
                    summary_writer_test.add_summary(summary, step)
                else:
                    summary, result = self.train_net.train_step_with_summary(sess, blob)
                    summary_writer_train.add_summary(summary, step)
                if is_drawing and self.display is not None:
                    if self.cfg.NET == 'ppn':
                        result['dim1'] = self.train_net.dim1
                        result['dim2'] = self.train_net.dim2
                    self.display(blob, self.cfg, index=step, name='display_train', directory=os.path.join(self.cfg.DISPLAY_DIR, 'train'), **result)

            if step%1000 == 0:
                save_path = saver.save(sess, os.path.join(self.outputdir, "model-%d.ckpt" % step))

        summary_writer_train.close()
        summary_writer_test.close()
        print("Done.")

def train_ppn(cfg):
    # Define data generators
    if cfg.TOYDATA:
        train_data = ToydataGenerator(cfg)
        test_data = ToydataGenerator(cfg)
    else:
        filelist = get_filelist(cfg.DATA)
        train_data = LarcvGenerator(cfg, ioname="train", filelist=filelist)
        test_data = LarcvGenerator(cfg, ioname="test", filelist=filelist)

    net_args = {"base_net": basenets[cfg.BASE_NET], "base_net_args": {}}

    if cfg.NET == 'ppn':
        display_util = display
    else:
        display_util = display_ppn_uresnet
    t = Trainer(PPN, train_data, test_data, cfg, display_util=display_util)
    t.train(net_args)

def train_classification(cfg):
    # Define data generators
    if cfg.TOYDATA:
        train_data = ToydataGenerator(cfg, classification=True)
        test_data = ToydataGenerator(cfg, classification=True)
    else:
        filelist = get_filelist(cfg.DATA)
        train_data = LarcvGenerator(cfg, ioname="train", filelist=filelist)
        test_data = LarcvGenerator(cfg, ioname="test", filelist=filelist)

    net_args = {}

    display_util = None
    if cfg.BASE_NET == 'uresnet':
        display_util = display_uresnet
    t = Trainer(basenets[cfg.BASE_NET], train_data, test_data, cfg, display_util=display_util)
    t.train(net_args, scope=cfg.BASE_NET)

def train_small_uresnet(cfg):
    filelist = get_filelist(cfg.DATA)
    train_data = LarcvGenerator(cfg, ioname="train", filelist=filelist)
    test_data = LarcvGenerator(cfg, ioname="test", filelist=filelist)
    t = Trainer(basenets[cfg.BASE_NET], train_data, test_data, cfg, display_util=display_uresnet)
    net_args = {"N": cfg.CROP_SIZE}
    t.train(net_args, scope="small_uresnet")

if __name__ == '__main__':
    #train_ppn(cfg)
    #train_classification(cfg)
    print(get_filelist("ls /data/drinkingkazu/dlprod_ppn_v05/ppn_p[01]*.root"))
