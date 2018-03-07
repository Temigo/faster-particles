# *-* encoding: utf-8 *-*
# Usage: python train_ppn.py i max_steps [vgg.ckpt vgg_16 0]
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import sys, os

from faster_particles.ppn import PPN
from faster_particles import ToydataGenerator
from faster_particles.demo_ppn import display
from faster_particles.base_net import VGG
#from config import cfg

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

        self.MAX_STEPS = cfg.MAX_STEPS
        self.weights_file = cfg.WEIGHTS_FILE
        self.display = display_util
        self.cfg = cfg

    def load_weights(self, sess, base_net="vgg"):
        # Restore variables for base net if given checkpoint file
        if self.weights_file is not None:
            print(tf.global_variables())
            variables_to_restore = [v for v in tf.global_variables() if base_net in v.name]
            saver_base_net = tf.train.Saver(variables_to_restore)
            saver_base_net.restore(sess, self.weights_file)

    def train(self, net_args):
        print("Creating net architecture...")
        net_args['cfg'] = self.cfg
        print(net_args, self.net)
        self.train_net = self.net(**net_args)
        self.test_net = self.net(**net_args)
        self.test_net.restore_placeholder(self.train_net.init_placeholders())
        self.train_net.create_architecture(is_training=True, reuse=False)
        self.test_net.create_architecture(is_training=False, reuse=True)
        print("Done.")

        #with tf.Session() as sess:
        sess = tf.InteractiveSession()

        self.load_weights(sess)
        summary_writer_train = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
        summary_writer_test = tf.summary.FileWriter(self.logdir + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())

        # Global saver
        saver = tf.train.Saver()

        step = 0
        while step < self.MAX_STEPS+1:
            step += 1
            is_testing = step%10 == 5
            is_drawing = step%1000 == 0
            if is_testing:
                blob = self.test_toydata.forward()
            else:
                blob = self.train_toydata.forward()

            if step%100 == 0:
                print("Step %d" % step)

            if is_testing:
                summary, result = self.test_net.test_image(sess, blob)
                summary_writer_test.add_summary(summary, step)
            else:
                summary, result = self.train_net.train_step_with_summary(sess, blob)
                summary_writer_train.add_summary(summary, step)
            if is_drawing and self.display is not None:
                self.display(blob, self.cfg, index=step, name='display_train', **result)

            if step%1000 == 0:
                save_path = saver.save(sess, self.outputdir + "/model-%d.ckpt" % step)

        summary_writer_train.close()
        summary_writer_test.close()

def train_ppn(cfg):
    # Define data generators
    train_toydata = ToydataGenerator(cfg)
    test_toydata = ToydataGenerator(cfg)
    net_args = {"base_net": VGG, "base_net_args": {}}

    t = Trainer(PPN, train_toydata, test_toydata, cfg, display_util=display)
    t.train(net_args)

def train_classification(cfg):
    # Define data generators
    train_toydata = ToydataGenerator(cfg, classification=True)
    test_toydata = ToydataGenerator(cfg, classification=True)
    net_args = {}

    t = Trainer(VGG, train_toydata, test_toydata, cfg)
    t.train(net_args)

if __name__ == '__main__':
    #train_ppn(cfg)
    #train_classification(cfg)
    pass
