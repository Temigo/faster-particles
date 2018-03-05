# *-* encoding: utf-8 *-*
# Usage: python train_ppn.py i max_steps [vgg.ckpt vgg_16]
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
from ppn_utils import build_vgg

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if not os.path.isdir("log"):
    os.makedirs("log")
if not os.path.isdir("display"):
    os.makedirs("display")

logdir = "log/run%d" % int(sys.argv[1])
outputdir = "/data/ldomine/run%d" % int(sys.argv[1])
MAX_STEPS = int(sys.argv[2])
weights_file = None
if len(sys.argv) == 4:
    weights_file = sys.argv[3]
    base_net = sys.argv[4]

# Define data generators
train_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
test_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)

train_net = PPN(build_base_net=build_vgg)
train_net.create_architecture(is_training=True, reuse=False)
test_net = PPN(build_base_net=build_vgg)
test_net.create_architecture(is_training=False, reuse=True)

#with tf.Session() as sess:
sess = tf.InteractiveSession()

summary_writer_train = tf.summary.FileWriter(logdir + '/train', sess.graph)
summary_writer_test = tf.summary.FileWriter(logdir + '/test', sess.graph)
sess.run(tf.global_variables_initializer())

step = 0
# Restore variables for base net if given checkpoint file
if weights_file is not None:
    variables_to_restore = [v for v in tf.global_variables() if base_net in v.name]
    saver_base_net = tf.train.Saver(variables_to_restore)
    saver_base_net.restore(sess, weights_file)
# Global saver
saver = tf.train.Saver()

while step < MAX_STEPS+1:
    step += 1
    is_testing = step%10 == 5
    is_drawing = step%1000 == 0
    if is_testing:
        blob = test_toydata.forward()
    else:
        blob = train_toydata.forward()

    if step%100 == 0:
        print("Step %d" % step)
    #print(blob['gt_pixels'])

    if is_testing:
        im_proposals, im_labels, im_scores, ppn1_proposals, rois, ppn2_proposals = test_net.test_image(sess, blob)
    else:
        summary, ppn1_proposals, labels_ppn1, rois, ppn2_proposals, ppn2_positives = train_net.train_step_with_summary(sess, blob, None)
        if is_drawing:
            display(blob, ppn1_labels=labels_ppn1, rois=rois, ppn2_proposals=ppn2_proposals, ppn2_positives=ppn2_positives, index=step, name='display_train')
    if is_testing:
        summary_writer_test.add_summary(summary, step)
    else:
        summary_writer_train.add_summary(summary, step)

    if step%1000 == 0:
        save_path = saver.save(sess, outputdir + "/model-%d.ckpt" % step)

summary_writer_train.close()
summary_writer_test.close()
