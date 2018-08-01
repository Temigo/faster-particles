import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from base_net import BaseNet

class UResNet(BaseNet):
    def __init__(self, cfg, N=0):
        super(UResNet, self).__init__(cfg, N=N)
        self.fn_conv = slim.conv2d
        self.fn_conv_transpose = slim.conv2d_transpose
        if self.is_3d:
            self.fn_conv = slim.conv3d
            self.fn_conv_transpose = slim.conv3d_transpose

        self.conv_feature_map = {}
        self.base_num_outputs = 16
        self._num_strides = 3#4

    def init_placeholders(self, image=None, labels=None):
        if self.is_3d:
            self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.N, self.N, self.N, 1), name="image_uresnet") if image is None else image
            self.pixel_labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.N, self.N, self.N), name="image_label") if labels is None else labels
        else:
            self.image_placeholder = tf.placeholder(tf.float32, shape=(None, self.N, self.N, 1), name="image_uresnet") if image is None else image
            self.pixel_labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.N, self.N), name="image_label") if labels is None else labels
        self.learning_rate_placeholder = tf.placeholder(tf.float32, name="lr")
        return [("image_placeholder", "image_uresnet"), ("learning_rate_placeholder", "lr"), ("pixel_labels_placeholder", "image_label")]

    def feed_dict(self, blob):
        return { self.image_placeholder: blob['data'], self.pixel_labels_placeholder: blob['labels'], self.learning_rate_placeholder: self.learning_rate }

    def test_image(self, sess, blob):
        predictions, summary = sess.run([self._predictions, self.summary_op], feed_dict=self.feed_dict(blob))
        return summary, {'predictions': predictions}

    def train_step_with_summary(self, sess, blobs):
        _, summary, predictions = sess.run([self.train_op, self.summary_op, self._predictions], feed_dict=self.feed_dict(blobs))
        return summary, {'predictions': predictions}

    def resnet_module(self, input_tensor, num_outputs, trainable=True, kernel=(3,3), stride=1, scope='noscope'):
        num_inputs  = input_tensor.get_shape()[-1].value
        with tf.variable_scope(scope):
            #
            # shortcut path
            #
            shortcut = None
            if num_outputs == num_inputs and stride ==1 :
                shortcut = input_tensor
            else:
                shortcut = self.fn_conv(inputs      = input_tensor,
                                   num_outputs = num_outputs,
                                   kernel_size = 1,
                                   stride      = stride,
                                   trainable   = trainable,
                                   padding     = 'same',
                                   normalizer_fn = slim.batch_norm,
                                   activation_fn = None,
                                   scope       = 'shortcut')
            #
            # residual path
            #
            residual = input_tensor
            residual = self.fn_conv(inputs      = residual,
                               num_outputs = num_outputs,
                               kernel_size = kernel,
                               stride      = stride,
                               trainable   = trainable,
                               padding     = 'same',
                               normalizer_fn = slim.batch_norm,
                               activation_fn = None,
                               scope       = 'resnet_conv1')

            residual = self.fn_conv(inputs      = residual,
                               num_outputs = num_outputs,
                               kernel_size = kernel,
                               stride      = 1,
                               trainable   = trainable,
                               padding     = 'same',
                               normalizer_fn = slim.batch_norm,
                               activation_fn = None,
                               scope       = 'resnet_conv2')

            return tf.nn.relu(shortcut + residual)

    def double_resnet(self, input_tensor, num_outputs, trainable=True, kernel=3, stride=1, scope='noscope'):
        with tf.variable_scope(scope):
            resnet1 = self.resnet_module(input_tensor=input_tensor,
                                    trainable=trainable,
                                    kernel=kernel,
                                    stride=stride,
                                    num_outputs=num_outputs,
                                    scope='module1')
            resnet2 = self.resnet_module(input_tensor=resnet1,
                                    trainable=trainable,
                                    kernel=kernel,
                                    stride=1,
                                    num_outputs=num_outputs,
                                    scope='module2')
        return resnet2

    def build_base_net(self, image_placeholder, is_training=True, reuse=False, scope="uresnet"):
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([self.fn_conv, slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                trainable=is_training):
                net = self.fn_conv(
                    inputs = image_placeholder,
                    num_outputs = self.base_num_outputs,
                    kernel_size = 7,
                    stride = 1,
                #    activation_fn = tf.nn.relu, FIXME this is default?
                    padding = 'same',
                    scope = 'conv0')
                self.conv_feature_map[net.get_shape()[-1].value] = net
                # Encoding steps
                for step in xrange(self._num_strides):
                    net = self.double_resnet(input_tensor = net,
                                        num_outputs  = net.get_shape()[-1].value * 2,
                                        kernel       = 3,
                                        stride       = 2,
                                        scope        = 'resnet_module%d' % step)
                    self.conv_feature_map[net.get_shape()[-1].value] = net

        keys = np.sort(self.conv_feature_map.keys())
        key2 = keys[-1]
        key = keys[int(len(keys)/2.0)-1] # -1
        print(self.conv_feature_map[key], self.conv_feature_map[key2])
        print(self.conv_feature_map)
        return self.conv_feature_map[key], self.conv_feature_map[key2]

    def create_architecture(self, is_training=True, reuse=False, scope="uresnet"):
        self.is_training = is_training
        self.reuse = reuse

        with slim.arg_scope([self.fn_conv, self.fn_conv_transpose, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            trainable=is_training):
            _, net = self.build_base_net(self.image_placeholder, is_training=is_training, reuse=reuse, scope=scope)
            with tf.variable_scope(scope, reuse=self.reuse):
                # Decoding steps
                for step in xrange(self._num_strides):
                    num_outputs = net.get_shape()[-1].value / 2
                    if not self.is_3d:
                        net = self.fn_conv_transpose(inputs      = net,
                                                num_outputs = num_outputs,
                                                kernel_size = 3,
                                                stride      = 2,
                                                padding     = 'same',
                                                scope       = 'deconv%d' % step)
                    else:
                        net = self.fn_conv_transpose(inputs      = net,
                                                num_outputs = num_outputs,
                                                kernel_size = 3,
                                                stride      = 2,
                                                padding     = 'same',
                                                scope       = 'deconv%d' % step,
                                                biases_initializer = None)

                    net = tf.concat([net, self.conv_feature_map[num_outputs]],
                                    axis=len(net.shape)-1,
                                    name='concat%d' % step)
                    net = self.double_resnet(input_tensor = net,
                                        num_outputs  = num_outputs,
                                        kernel       = 3,
                                        stride       = 1,
                                        scope        = 'resnet_module%d' % (step+5))

                # Final conv layers
                net = self.fn_conv(inputs      = net,
                              num_outputs = self.base_num_outputs,
                              padding     = 'same',
                              kernel_size = 7,
                              stride      = 1,
                              scope       = 'conv1')
                net = self.fn_conv(inputs      = net,
                              num_outputs = self.num_classes,
                              padding     = 'same',
                              kernel_size = 3,
                              stride      = 1,
                              activation_fn = None,
                              scope = 'conv2')

                self._softmax = tf.nn.softmax(logits=net)
                self._predictions = tf.argmax(self._softmax, axis=-1)

                # Define loss
                dims = self.image_placeholder.get_shape()[1:]

                if is_training:
                    with tf.variable_scope('metrics'):
                        labels = tf.argmax(net, axis=len(dims), output_type=tf.int32)
                        self.accuracy_allpix = tf.reduce_mean(tf.cast(tf.equal(labels, self.pixel_labels_placeholder),tf.float32))
                        nonzero_idx = tf.where(tf.reshape(self.image_placeholder, tf.shape(self.image_placeholder)[:-1]) > tf.to_float(0.))
                        nonzero_label = tf.gather_nd(self.pixel_labels_placeholder, nonzero_idx)
                        nonzero_pred  = tf.gather_nd(labels, nonzero_idx)
                        self.accuracy_nonzero = tf.reduce_mean(tf.cast(tf.equal(nonzero_label, nonzero_pred), tf.float32))
                        tf.summary.scalar('accuracy_all', self.accuracy_allpix)
                        tf.summary.scalar('accuracy_nonzero', self.accuracy_nonzero)
                    self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.pixel_labels_placeholder, logits=net)
                    self._loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(self._loss, [-1, int(np.prod(dims) / dims[-1])]), axis=1))
                    tf.summary.scalar('loss', self._loss)

                    self.train_op = tf.train.AdamOptimizer(self.learning_rate_placeholder).minimize(self._loss)

                # Define summary
                self.summary_op = tf.summary.merge_all()

if __name__ == '__main__':
    class config(object):
        IMAGE_SIZE = 512
        NUM_CLASSES = 3
        LEARNING_RATE = 0.001
        DATA_3D = True
    v = UResNet(cfg=config())
    print(dir(v))
    v.init_placeholders()
    v.create_architecture()
