# *-* encoding: utf-8 *-*
# Simple RPN taken from Faster-RCNN

import numpy as np
import tensorflow as tf

class RPN(object):
    def __init__(self):
        self._num_classes = 4
        #self._train_weight_decay = # FIXME
        self._image = None
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._scope = 'faster_particles'
        self._learning_rate = 0.001 # FIXME

    def set_input_shape(self,tensor):
        try:
            self._input_shape = tf.shape(tensor)
        except Exception:
            self._input_shape = np.shape(tensor)

    def train(self, num_classes, train_io, trainable=True):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 1])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])

        self._num_classes = num_classes
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005) # Train weight decay
        biases_regularizer = tf.no_regularizer
        # FIXME = weights_regularizer if use bias_decay?

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer,
                    biases_initializer=tf.constant_initializer(0.0)):
            #rois, cls_prob, bbox_pred = self._build_network(trainable=training)
            # select initializers
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            net = None
            with tf.variable_scope(self._scope, self._scope):
                # Image to head layers - FIXME based on VGG-16 model
                net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                                  trainable=False, scope='conv1')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                                trainable=False, scope='conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                                trainable=trainable, scope='conv3')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                trainable=trainable, scope='conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                trainable=trainable, scope='conv5')

                #self._region_proposal(net, trainable, initializer)
                # Step of operations:
                # 0) Convolution layer ... shared by RPN and detection
                # 1) Two parallel convolution layers ... 4 region proposals (bbox) and 2 object-ness classification (cls)
                # 2) Reshaping + softmax + re-reshaping to get candidate ROIs FIXME
                # 3) Select a sub-set of ROIs and scores from proposal_layer

                # Step 0) Convolution for RPN/Detection shared layer
                rpn = slim.conv2d(net, 512, # RPN Channels for VGG-16 (intermediate layer)
                                  [3, 3],
                                  trainable=trainable,
                                  weights_initializer=initializer,
                                  scope="rpn_conv/3x3")
                # Step 1-a) RPN 4 bbox prediction parameters
                rpn_bbox_pred = slim.conv2d(rpn, 4, [1, 1],
                                            trainable=trainable,
                                            weights_initializer=initializer,
                                            padding='VALID',
                                            activation_fn=None,
                                            scope='rpn_bbox_pred')
                # Step 1-b) Generate 2 class scores
                rpn_cls_score = slim.conv2d(rpn, 2, [1, 1],
                                            trainable=trainable,
                                            weights_initializer=initializer,
                                            padding='VALID',
                                            activation_fn=None,
                                            scope='rpn_cls_score')
                # Step 2-b) Compute softmax
                rpn_cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(rpn_cls_score,
                                                                           [-1, tf.shape(rpn_cls_score)[-1]]),
                                                                name='rpn_cls_prob'),
                                                  tf.shape(rpn_cls_score))
                # TODO Derive ROIs and ROIs scores using NMS and ground truth
                self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
                self._predictions["rpn_cls_score"] = rpn_cls_score

        # add train summary
        #for var in tf.trainable_variables():
        #    self._train_summaries.append(var)

        layers_to_output = {}

        if testing:
            #stds = np.tile(np.array(self._cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            #means = np.tile(np.array(self._cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            #self._predictions["bbox_pred"] *= stds
            #self._predictions["bbox_pred"] += means
        else:
            # self._add_losses()
            # For now only RPN losses - FIXME RNN losses

            # RPN, class loss
            # Object present/absent classification (# channel =2)
            rpn_cls_score = self._predictions['rpn_cls_score']
            # Object present/absent label for _every_ anchor
            rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            # Get rpn_bbox_pred from region_proposal
            # ... which is a prediction of objecxt location based on the anchor location
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            # Get rpn_bbox_targets from anchor_target_layer
            # (which is "true distance" of an anchor to a corresponding truth box location)
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,

            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            loss = rpn_cross_entropy + rpn_loss_box
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss + regularization_loss

            layers_to_output.update(self._losses)

        layers_to_output.update(self._predictions)

        # Define the loss
        loss = layers_to_output['total_loss']
        # Set learning rate and momentum
        lr = tf.Variable(self._learning_rate, trainable=False)
        self.optimizer = tf.train.MomentumOptimizer(lr, 0.9) # Train Momentum

        # Compute the gradients with regard to the loss
        gvs = self.optimizer.compute_gradients(loss)
        # FIXME double bias
        train_op = self.optimizer.apply_gradients(gvs)

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        with tf.Session(config=tfconfig) as sess:
            for iter <= max_iters:
                # Train step
                blobs = train_io.forward()
                feed_dict = {
                    self._image: blobs['data'],
                    self._input_shape: blobs['im_info'],
                    self._gt_boxes: blobs['gt_boxes']
                    }
                _, rpn_cls_score, rpn_bbox_pred = sess.run([train_op,
                        self._predictions["rpn_cls_score"],
                        self._predictions["rpn_bbox_pred"]
                        ], feed_dict=feed_dict)

                print rpn_cls_score.shape
                print rpn_bbox_pred.shape

if __name__ == '__main__':
    rpn = RPN()
    train_io = ToydataGenerator(128, 5, 1)
    rpn.train(4, train_io)
