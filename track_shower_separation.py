# *-* encoding: utf-8
# Train VGG for track/shower separation
# Usage: python track_shower_separation.py logdir outputdir

import tensorflow as tf
import tensorflow.contrib.slim as slim
from toydata_generator import ToydataGenerator

import sys, time, os
import numpy as np

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

class ClassificationNetwork(object):
    N = 256 # Size of square image
    num_classes = 3
    max_tracks = 1
    max_kinks = 3
    max_steps = 20000
    learning_rate = 0.001
    logdir = sys.argv[1]
    outputdir = sys.argv[2]
    max_track_length = 200

    def __init__(self):
        # Define placeholders
        self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.N, self.N, 3))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1))
        self.learning_rate_placeholder = tf.placeholder(tf.float32)

    def build_network(self):
        # Define network
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        biases_regularizer = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=weights_regularizer, biases_regularizer=biases_regularizer, biases_initializer=tf.constant_initializer(0.0)):
            self.net = slim.repeat(self.image_placeholder, 1, slim.conv2d, 64, [3, 3], trainable=True, scope='conv1', stride=2)
            self.net = slim.max_pool2d(self.net, [2, 2], padding='SAME', scope='pool1')
            self.net = slim.repeat(self.net, 1, slim.conv2d, 128, [3, 3], trainable=True, scope='conv2')
            self.net = slim.max_pool2d(self.net, [2, 2], padding='SAME', scope='pool2')
            self.net = slim.repeat(self.net, 1, slim.conv2d, 256, [3, 3], trainable=True, scope='conv3')
            self.net = slim.max_pool2d(self.net, [2, 2], padding='SAME', scope='pool3')
            self.net = slim.repeat(self.net, 1, slim.conv2d, 512, [3, 3], trainable=True, scope='conv4')
            self.net = slim.max_pool2d(self.net, [2, 2], padding='SAME', scope='pool4')
            #self.net = slim.repeat(self.net, 3, slim.conv2d, 512, [3, 3], trainable=True, scope='conv5')
            #self.net = slim.max_pool2d(self.net, [2, 2], padding='SAME', scope='pool5')
            self.net_flat = slim.flatten(self.net, scope='flatten')
            self.fc6 = slim.fully_connected(self.net_flat, 4096, scope='fc6')
            self.fc6 = slim.dropout(self.fc6, keep_prob=0.5, is_training=True, scope='dropout6')
            #self.fc7 = slim.fully_connected(self.fc6, 4096, scope='fc7')
            #self.fc7 = slim.dropout(self.fc7, keep_prob=0.5, is_training=True, scope='dropout7')
            self.cls_score = slim.fully_connected(self.fc6, self.num_classes, weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True, activation_fn=None, scope='cls_score')
            self.cls_prob = tf.nn.softmax(self.cls_score, name='cls_prob')
            self.cls_pred = tf.argmax(self.cls_score, axis=1, name='cls_pred')

        # Define loss
        self.labels = tf.reshape(self.labels_placeholder, [-1])
        #print cls_score.shape, labels_placeholder.shape
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_score, labels=self.labels))
        #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate_placeholder).minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate_placeholder).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.cls_pred, tf.int32), self.labels), tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

        # Define summary
        self.summary_op = tf.summary.merge_all()

        # Define model saver
        self.saver = tf.train.Saver()

    def train(self):
        # Define data generators
        train_toydata = ToydataGenerator(self.N, self.max_tracks, self.max_kinks, max_track_length=self.max_track_length, classification=True)
        test_toydata = ToydataGenerator(self.N, self.max_tracks, self.max_kinks, max_track_length=self.max_track_length, classification=True)

        with tf.Session() as sess:
            summary_writer_train = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
            #summary_writer_train.add_graph(sess.graph)
            summary_writer_test = tf.summary.FileWriter(self.logdir + '/test', sess.graph)
            #summary_writer_test.add_graph(sess.graph)
            #if len(sys.argv) > 3:
            #    self.saver.restore(sess, sys.argv[3])

            data_generation_time = 0
            training_time = 0
            testing_time = 0
            summary_time = 0

            sess.run(tf.global_variables_initializer())
            for step in range(self.max_steps):
                #write_summary = step % 3 == 0
                write_summary = step%3==0
                save_model = step % 1000 == 0
                if step==10000:
                    self.learning_rate *= 0.1
                if step%5 ==0: # Test
                    print "Testing epoch %d" % step
                    start_time = time.time()
                    blob = test_toydata.forward()
                    data_time = time.time()
                    summary, _ = sess.run([self.summary_op, self.accuracy], feed_dict={self.image_placeholder: blob['data'], self.labels_placeholder: blob['image_label'], self.learning_rate_placeholder: self.learning_rate})
                    run_time = time.time()

                    data_generation_time += data_time - start_time
                    testing_time += run_time - data_time

                    if write_summary:
                        summary_writer_test.add_summary(summary, step)
                        summary_t = time.time()
                        summary_time += summary_t - run_time
                else:
                    print "Training epoch %d" % step
                    start_time = time.time()
                    blob = train_toydata.forward()
                    data_time = time.time()
                    summary, _ = sess.run([self.summary_op, self.train_op], feed_dict={self.image_placeholder: blob['data'], self.labels_placeholder: blob['image_label'], self.learning_rate_placeholder: self.learning_rate})
                    run_time = time.time()

                    data_generation_time += data_time - start_time
                    training_time += run_time - data_time

                    # tf.one_hot(blob['gt_labels'], num_classes)
                    if write_summary:
                        summary_writer_train.add_summary(summary, step)
                        summary_t = time.time()
                        summary_time += summary_t - run_time
                if save_model:
                    save_path = self.saver.save(sess, self.outputdir + "model-%d.ckpt" % step)
                    print "Model saved in %s" % save_path

            print "Data generation took %f\nTraining took %f\nTesting took %f\nSummary took %f\n" % (data_generation_time, training_time, testing_time, summary_time)
                    #if step%10 == 0: # Test
                    #    summary, _ = sess.run([summary, test_op])
            summary_writer_train.close()
            summary_writer_test.close()

    def plot_track_length(self):
        accuracy = []
        track_lengths = []
        with tf.Session() as sess:
            self.saver.restore(sess, sys.argv[1])
            for track_length in range(1, self.N):
                toydata = ToydataGenerator(self.N, self.max_tracks, 1, max_track_length=track_length, classification=True, seed=track_length)
                for i in range(100):
                    blob = toydata.forward()
                    while blob['image_label'][0][0] != 1:
                        blob = toydata.forward()
                    acc = sess.run(self.accuracy, feed_dict={self.image_placeholder: blob['data'], self.labels_placeholder: blob['image_label'], self.learning_rate_placeholder: self.learning_rate})
                    accuracy.append(acc)
                    track_lengths.append(blob['track_length'])

        np.savetxt("accuracy_track_length.csv", np.asarray(accuracy), delimiter=",")
        np.savetxt("track_lengths.csv", np.asarray(track_lengths), delimiter=",")
        #print accuracy, track_lengths

    def plot_kinks(self):
        accuracy = []
        accuracy_error = []
        kinks = []
        with tf.Session() as sess:
            self.saver.restore(sess, sys.argv[1])
            for max_kinks in range(1, 11):
                toydata = ToydataGenerator(self.N, self.max_tracks, max_kinks, max_track_length=self.max_track_length, classification=True, seed=max_kinks, kinks=max_kinks)
                acc_kink = []
                for i in range(1000):
                    blob = toydata.forward()
                    #while blob['image_label'][0][0] != 1:
                    #    blob = toydata.forward()
                    acc = sess.run(self.accuracy, feed_dict={self.image_placeholder: blob['data'], self.labels_placeholder: blob['image_label'], self.learning_rate_placeholder: self.learning_rate})
                    acc_kink.append(acc)
                    #kinks.append(blob['kinks'])
                kinks.append(max_kinks)
                accuracy.append(np.mean(acc_kink))
                accuracy_error.append(np.std(acc_kink))

        np.savetxt("accuracy_kinks.csv", np.asarray(accuracy), delimiter=",")
        np.savetxt("accuracy_kinks_error.csv", np.asarray(accuracy_error), delimiter=",")
        np.savetxt("kinks.csv", np.asarray(kinks), delimiter=",")


    def inference(self):
        self.max_kinks = 10
        toydata = ToydataGenerator(self.N, self.max_tracks, self.max_kinks, max_track_length=self.max_track_length, classification=True, seed=0)
        with tf.Session() as sess:
            self.saver.restore(sess, sys.argv[1])
            for i in range(10):
                blob = toydata.forward()
                print blob
                cls_pred = sess.run(self.cls_pred, feed_dict={self.image_placeholder: blob['data'], self.labels_placeholder: blob['image_label'], self.learning_rate_placeholder: self.learning_rate})
                print blob['image_label'], cls_pred

        #fig, ax = plt.subplots(figsize=(12, 12), facecolor='w')
        #ax.imshow(blob['data'][0,:,:,0], interpolation='none', cmap='jet', origin='lower')
        #plt.draw()
        #fig.savefig('out_inference.png')


if __name__ == '__main__':
    c = ClassificationNetwork()
    c.build_network()
    if sys.argv[-1] == 'train':
        c.train()
    elif sys.argv[-1] == 'inference':
        c.inference()
    elif sys.argv[-1] == 'track-length':
        c.plot_track_length()
    elif sys.argv[-1] == 'kink':
        c.plot_kinks()
