# *-* encoding: utf-8 *-*
# Usage: python train_ppn.py i max_steps
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys, os

from ppn import PPN
from toydata_generator import ToydataGenerator

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

logdir = "log/run%d" % int(sys.argv[1])
outputdir = "output/run%d" % int(sys.argv[1])
MAX_STEPS = int(sys.argv[2])
# Define data generators
train_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
test_toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)

net = PPN()
net.create_architecture()
saver = tf.train.Saver()

with tf.Session() as sess:
    summary_writer_train = tf.summary.FileWriter(logdir + '/train', sess.graph)
    summary_writer_test = tf.summary.FileWriter(logdir + '/test', sess.graph)
    sess.run(tf.global_variables_initializer())
    for step in range(MAX_STEPS):
        is_testing = step%10 == 5
        if is_testing:
            blob = test_toydata.forward()
        else:
            blob = train_toydata.forward()

        print("Step %d" % step)
        print(blob['gt_pixels'])

        if is_testing:
            net.test_image(blob)
        else:
            summary = net.train_step_with_summary(sess, blob, None)

        if is_testing:
            summary_writer_test.add_summary(summary, step)
        else:
            summary_writer_train.add_summary(summary, step)

        if step%1000 == 0:
            save_path = saver.save(sess, outputdir + "/model-%d.ckpt" % step)

    summary_writer_train.close()
    summary_writer_test.close()
