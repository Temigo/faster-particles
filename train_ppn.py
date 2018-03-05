# *-* encoding: utf-8 *-*
# Usage: python train_ppn.py i max_steps [vgg.ckpt vgg_16 0]
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys, os
import matplotlib
matplotlib.use('Agg')

from ppn import PPN
from toydata_generator import ToydataGenerator
from demo_ppn import display
from base_net import VGG

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Trainer(object):
    def __init__(self, net, train_toydata, test_toydata, display_util=None, max_steps=100, weights_file=None, logdir="log", displaydir="display", outputdir="output"):
        self.train_toydata = train_toydata
        self.test_toydata = test_toydata
        self.net = net
        self.logdir = logdir
        self.displaydir = displaydir
        self.outputdir = outputdir
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.isdir(self.displaydir):
            os.makedirs(self.displaydir)
        if not os.path.isdir(self.outputdir):
            os.makedirs(self.outputdir)

        self.MAX_STEPS = max_steps
        self.weights_file = weights_file
        self.display = display_util

    def load_weights(self, sess, base_net=""):
        # Restore variables for base net if given checkpoint file
        if self.weights_file is not None:
            print(tf.global_variables())
            variables_to_restore = [v for v in tf.global_variables() if base_net in v.name]
            saver_base_net = tf.train.Saver(variables_to_restore)
            saver_base_net.restore(sess, weights_file)

    def train(self, net_args):
        print("Creating net architecture...")
        self.train_net = self.net(**net_args)
        self.test_net = self.net(**net_args)
        self.train_net.create_architecture(is_training=True, reuse=False)
        self.test_net.create_architecture(is_training=False, reuse=True)
        print("Done.")

        #with tf.Session() as sess:
        sess = tf.InteractiveSession()

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
            #print(blob['gt_pixels'])

            if is_testing:
                summary, result = self.test_net.test_image(sess, blob)
                summary_writer_test.add_summary(summary, step)
            else:
                summary, result = self.train_net.train_step_with_summary(sess, blob)
                summary_writer_train.add_summary(summary, step)
            if is_drawing and self.display is not None:
                self.display(blob, index=step, name='display_train', **result)

            if step%1000 == 0:
                save_path = saver.save(sess, self.outputdir + "/model-%d.ckpt" % step)

        summary_writer_train.close()
        summary_writer_test.close()

def train_ppn():
    # Define data generators
    train_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
    test_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
    net_args = {"base_net": VGG}

    t = Trainer(PPN, train_toydata, test_toydata, display_util=display, max_steps=100, logdir="log/run17", displaydir="display/run17", outputdir="/data/ldomine/run17")
    t.train(net_args)

def train_classification():
    # Define data generators
    train_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200, classification=True)
    test_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200, classification=True)
    net_args = {"N": 512, "num_classes": 3}

    t = Trainer(VGG, train_toydata, test_toydata, logdir="log/vgg", displaydir="display/vgg", outputdir="/data/ldomine/vgg")
    t.train(net_args)

if __name__ == '__main__':
    #train_ppn()
    train_classification()
