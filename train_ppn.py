# *-* encoding: utf-8 *-*
# Usage: python train_ppn.py i max_steps [vgg.ckpt]
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

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logdir = "log/run%d" % int(sys.argv[1])
outputdir = "/data/ldomine/run%d" % int(sys.argv[1])
MAX_STEPS = int(sys.argv[2])
# Define data generators
train_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
test_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)

train_net = PPN()
train_net.create_architecture(is_training=True, reuse=False)
test_net = PPN()
test_net.create_architecture(is_training=False, reuse=True)
saver = tf.train.Saver()

#with tf.Session() as sess:
sess = tf.InteractiveSession()
# saver.restore(sess, sys.argv[3])
# FIXME Fix variables? Cf faster-rcnn trainer.py source code
summary_writer_train = tf.summary.FileWriter(logdir + '/train', sess.graph)
summary_writer_test = tf.summary.FileWriter(logdir + '/test', sess.graph)
sess.run(tf.global_variables_initializer())
for step in range(MAX_STEPS):
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
