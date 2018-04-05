# Define some base networks for PPN
# Currently only VGG is implemented.
# To add more base networks inherit from BaseNet class and update basenets variable.

import tensorflow as tf
import tensorflow.contrib.slim as slim


class BaseNet(object):
    """
    Skeleton class from which any base network should inherit.
    PPN will assume this class structure for the base network.
    """
    def __init__(self, cfg):
        self.N = cfg.IMAGE_SIZE
        self.num_classes = cfg.NUM_CLASSES
        self.learning_rate = 0.001
        self.is_3d = cfg.DATA_3D

    def init_placeholders(self):
        """
        Define placeholders here as class attributes.
        @return: List of tuples (attribute_name, tf_name)
        """
        raise NotImplementedError

    def restore_placeholder(self, names):
        """
        For the test network to restore the placeholders previously created
        by the train network.
        @param names List of tuples (attribute_name, tf_name)
        """
        raise NotImplementedError

    def test_image(self, sess, blob):
        """
        Run inference on a single image.
        @param sess Tensorflow current session.
        @param blob Data dictionary.
        @return: summary, dictionary of results
        """
        raise NotImplementedError

    def train_step_with_summary(self, sess, blobs):
        """
        Run training step.
        @param sess Tensorflow current session.
        @param blob Data dictionary.
        @return: summary, dictionary of results
        """
        raise NotImplementedError

    def build_base_net(self, image_placeholder, is_training=True, reuse=False):
        """
        Called by PPN and by the base net during training.
        @return: F3 and F5 layers
        """
        raise NotImplementedError

    def create_architecture(self, is_training=True, reuse=False, scope="base_net"):
        """
        Called to train the base network only.
        @param scope Scope name to be shared between base network and PPN.
        """
        raise NotImplementedError


class VGG(BaseNet):
    def init_placeholders(self):
        if self.is_3d:
            self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.N, self.N, self.N, 1), name="image")
        else:
            self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.N, self.N, 1), name="image")
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name="labels")
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name="lr")
        return [("image_placeholder", "image"), ("labels_placeholder", "labels"), ("learning_rate_placeholder", "lr")]

    def restore_placeholder(self, names):
        for attr, name in names:
            setattr(self, attr, tf.get_default_graph().get_tensor_by_name(name + ':0'))

    def test_image(self, sess, blob):
        feed_dict = { self.image_placeholder: blob['data'], self.labels_placeholder: blob['image_label'], self.learning_rate_placeholder: self.learning_rate }
        acc, cls_pred, summary = sess.run([self.accuracy, self.cls_pred, self.summary_op], feed_dict=feed_dict)
        return summary, {'acc': acc, 'cls_pred': cls_pred}

    def train_step_with_summary(self, sess, blobs):
        feed_dict = { self.image_placeholder: blobs['data'], self.labels_placeholder: blobs['image_label'], self.learning_rate_placeholder: self.learning_rate }
        _, summary = sess.run([self.train_op, self.summary_op], feed_dict=feed_dict)
        return summary, {}

    def build_base_net(self, image_placeholder, is_training=True, reuse=False):
        # =====================================================
        # --- VGG16 net = 13 conv layers with 5 max-pooling ---
        # =====================================================
        with tf.variable_scope("vgg_16", reuse=reuse):
            weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
            biases_regularizer = tf.no_regularizer
            conv = slim.conv2d
            max_pool = slim.max_pool2d
            dim = 2
            num_channels = 64
            if self.is_3d:
                conv = slim.conv3d
                max_pool = slim.max_pool3d
                dim = 3
                num_channels = 2

            with slim.arg_scope([conv, slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                trainable=is_training,
                                weights_regularizer=weights_regularizer,
                                biases_regularizer=biases_regularizer,
                                biases_initializer=tf.constant_initializer(0.0)):
                net = slim.repeat(image_placeholder, 2, conv, num_channels, [3,] * dim,
                                  trainable=is_training, scope='conv1')
                net = max_pool(net, [2,] * dim, padding='SAME', scope='pool1')
                net = slim.repeat(net, 2, conv, 2 * num_channels, [3,] * dim,
                                trainable=is_training, scope='conv2')
                net = max_pool(net, [2,] * dim, padding='SAME', scope='pool2')
                net = slim.repeat(net, 3, conv, 4 * num_channels, [3,] * dim,
                                trainable=is_training, scope='conv3')
                net = max_pool(net, [2,] * dim, padding='SAME', scope='pool3')
                net2 = slim.repeat(net, 3, conv, 8 * num_channels, [3,] * dim,
                                trainable=is_training, scope='conv4')
                net2 = max_pool(net2, [2,] * dim, padding='SAME', scope='pool4')
                net2 = slim.repeat(net2, 3, conv, 8 * num_channels, [3,] * dim,
                                trainable=is_training, scope='conv5')
                net2 = max_pool(net2, [2,] * dim, padding='SAME', scope='pool5')
                # After 5 times (2, 2) pooling, if input image is 512x512
                # the feature map should be spatial dimensions 16x16.
            return net, net2

    def create_architecture(self, is_training=True, reuse=False, scope="vgg_full"):
        self.is_training = is_training
        self.reuse = reuse
        # Define network
        with tf.variable_scope(scope, reuse=self.reuse):
            weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
            biases_regularizer = tf.no_regularizer
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                weights_regularizer=weights_regularizer,
                biases_regularizer=biases_regularizer,
                biases_initializer=tf.constant_initializer(0.0)):

                _, self.net = self.build_base_net(self.image_placeholder, is_training=self.is_training, reuse=self.reuse)

                self.net_flat = slim.flatten(self.net, scope='flatten')
                self.fc6 = slim.fully_connected(self.net_flat, 2048, scope='fc6')
                self.fc6 = slim.dropout(self.fc6, keep_prob=0.5, is_training=self.is_training, scope='dropout6')
                #self.fc7 = slim.fully_connected(self.fc6, 4096, scope='fc7')
                #self.fc7 = slim.dropout(self.fc7, keep_prob=0.5, is_training=True, scope='dropout7')
                self.cls_score = slim.fully_connected(self.fc6, self.num_classes, weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True, activation_fn=None, scope='cls_score')
                self.cls_prob = tf.nn.softmax(self.cls_score, name='cls_prob')
                self.cls_pred = tf.argmax(self.cls_score, axis=1, name='cls_pred')
            # Define loss
            self.labels = tf.reshape(self.labels_placeholder, [-1])
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.cls_score, labels=self.labels))
            self.train_op = tf.train.AdamOptimizer(self.learning_rate_placeholder).minimize(self.loss)
            tf.summary.scalar('loss', self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.cls_pred, tf.int32), self.labels), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            # Define summary
            self.summary_op = tf.summary.merge_all()

# Available base networks
basenets = { "vgg": VGG }

if __name__ == '__main__':
    class config(object):
        IMAGE_SIZE = 512
        NUM_CLASSES = 3
        LEARNING_RATE = 0.001
    v = VGG(cfg=config())
    print(dir(v))
