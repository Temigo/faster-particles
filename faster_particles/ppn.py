# *-* encoding: utf-8 *-*
# Pixel Proposal Network
# Draft implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys, os

from faster_particles.ppn_utils import include_gt_pixels, compute_positives_ppn2, \
    compute_positives_ppn1, assign_gt_pixels, generate_anchors, \
    predicted_pixels, top_R_pixels, slice_rois, crop_pool_layer
from faster_particles.base_net import VGG

class PPN(object):

    def __init__(self, cfg=None, base_net=VGG, base_net_args={}):
        """
        Allow for easy implementation of different base network architecture:
        build_base_net should take as inputs
        (image_placeholder, is_training=True, reuse=False)
        and return (F3, F5) conv layers
        """
        # Global parameters
        self.R = cfg.R
        self.num_classes = cfg.NUM_CLASSES # (B)ackground, (T)rack edge, (S)hower start, (S+T)
        self.N = cfg.IMAGE_SIZE
        self.ppn1_score_threshold = cfg.PPN1_SCORE_THRESHOLD
        self.ppn2_distance_threshold = cfg.PPN2_DISTANCE_THRESHOLD
        self.lr = cfg.LEARNING_RATE # Learning rate
        self.lambda_ppn1 = cfg.LAMBDA_PPN1 # Balance loss between class and distance in ppn1
        self.lambda_ppn2 = cfg.LAMBDA_PPN2 # Balance loss between class and distance in ppn2
        self.lambda_ppn = cfg.LAMBDA_PPN # Balance loss between ppn1 and ppn2
        self._predictions = {}
        self._losses = {}
        self.base_net = base_net(cfg=cfg, **base_net_args)
        self.cfg = cfg

    def test_image(self, sess, blob):
        feed_dict = { self.image_placeholder: blob['data'], self.gt_pixels_placeholder: blob['gt_pixels'] }
        im_proposals, im_labels, im_scores, rois, summary = sess.run([
            self._predictions['im_proposals'],
            self._predictions['im_labels'],
            self._predictions['im_scores'],
            self._predictions['rois'],
            self.summary_op
            ], feed_dict=feed_dict)
        return summary, {'im_proposals': im_proposals,
                        'im_labels': im_labels,
                        'im_scores': im_scores,
                        'rois': rois}

    def train_step_with_summary(self, sess, blobs):
        feed_dict = { self.image_placeholder: blobs['data'], self.gt_pixels_placeholder: blobs['gt_pixels'] }
        _, ppn1_proposals, labels_ppn1, rois, ppn2_proposals, ppn2_positives, \
        im_labels, im_scores, im_proposals, summary = sess.run([
                            self.train_op,
                            self._predictions['ppn1_proposals'],
                            self._predictions['labels_ppn1'],
                            self._predictions['rois'],
                            self._predictions['ppn2_proposals'],
                            self._predictions['ppn2_positives'],
                            self._predictions['im_labels'],
                            self._predictions['im_scores'],
                            self._predictions['im_proposals'],
                            self.summary_op
                            ], feed_dict=feed_dict)

        print(blobs['gt_pixels'])
        print("#positives: ", np.sum(ppn2_positives))

        return summary, {'rois': rois,
                        'im_labels': im_labels,
                        'im_proposals': im_proposals,
                        'im_scores': im_scores}

    def init_placeholders(self):
        # Define placeholders
        # FIXME Assuming batch size of 1 currently
        if self.cfg.DATA_3D:
            self.image_placeholder       = tf.placeholder(name="image", shape=(1, self.N, self.N, self.N, 1), dtype=tf.float32)
            # Shape of gt_pixels_placeholder = nb_gt_pixels, 3 coordinates + 1 class label in [0, num_classes)
            self.gt_pixels_placeholder   = tf.placeholder(name="gt_pixels", shape=(None, 4), dtype=tf.float32)
        else:
            self.image_placeholder       = tf.placeholder(name="image", shape=(1, self.N, self.N, 1), dtype=tf.float32)
            # Shape of gt_pixels_placeholder = nb_gt_pixels, 2 coordinates + 1 class label in [0, num_classes)
            self.gt_pixels_placeholder   = tf.placeholder(name="gt_pixels", shape=(None, 3), dtype=tf.float32)
        return [("image_placeholder", "image"), ("gt_pixels_placeholder", "gt_pixels")]

    def restore_placeholder(self, names):
        for attr, name in names:
            setattr(self, attr, tf.get_default_graph().get_tensor_by_name(name + ':0'))

    def set_dimensions(self, f3shape, f5shape):
        f3shape = f3shape.as_list()
        f5shape = f5shape.as_list()
        self.N2 = f3shape[1]
        self.N3 = f5shape[1]
        #if self.N%self.N2 != 0 or self.N2%self.N3 != 0:
        #    raise Exception("Layers dimensions are incompatibles.")
        self.dim1 = int(self.N/self.N2)
        self.dim2 = int(self.N2/self.N3)
        print("N2 = %d ; N3 = %d ; dim1 = %d ; dim2 = %d" % (self.N2, self.N3, self.dim1, self.dim2))

    def set3d(self):
        self.conv = slim.conv2d
        self.dim = 2
        self.ppn1_channels, self.ppn2_channels = 512, 512
        if self.cfg.DATA_3D:
            self.conv = slim.conv3d
            self.dim = 3
            self.ppn1_channels, self.ppn2_channels = 16, 16

    def create_architecture(self, is_training=True, reuse=None, scope="ppn"):
        self.is_training = is_training
        self.reuse = reuse

        # Define network regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        biases_regularizer = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.conv3d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            trainable=self.is_training,
                            weights_regularizer=weights_regularizer,
                            biases_regularizer=biases_regularizer,
                            biases_initializer=tf.constant_initializer(0.0)):
            with tf.variable_scope(scope, reuse=self.reuse):
                # Returns F3 and F5 feature maps
                net, net2 = self.base_net.build_base_net(self.image_placeholder, is_training=self.is_training, reuse=self.reuse)
                self.set_dimensions(net.shape, net2.shape)
                self.set3d()

                # Build PPN1
                rois = self.build_ppn1(net2)
                rois = slice_rois(rois, self.dim2)

                if self.is_training:
                    # During training time, check if all ground truth pixels are covered by ROIs
                    # If not, add relevant ROIs on F3
                    rois = include_gt_pixels(rois, self.get_gt_pixels(), self.dim1, self.dim2)
                    assert rois.get_shape().as_list() == [None, self.dim]

                self._predictions['rois'] = rois

                # Pool to Pixels of Interest of intermediate layer
                # Shape of rpn_pooling = nb_rois, 1, 1, 256
                rpn_pooling = crop_pool_layer(net, rois, self.dim2, self.dim)

                proposals2, scores2 = self.build_ppn2(rpn_pooling, rois)

                # FIXME How to combine losses
                total_loss = tf.identity(self.lambda_ppn * (self.lambda_ppn1 * self._losses['loss_ppn1_point'] \
                            + (1.0 - self.lambda_ppn1) * self._losses['loss_ppn1_class']) \
                            + (1.0 - self.lambda_ppn) * (self.lambda_ppn2 * self._losses['loss_ppn2_point'] \
                            + (1.0 - self.lambda_ppn2) * self._losses['loss_ppn2_class']), name="total_loss")
                self._losses['total_loss'] = total_loss
                tf.summary.scalar('loss', total_loss)

                if self.is_training:
                    tf.summary.scalar('ppn1_positives', tf.reduce_sum(tf.cast(self._predictions['ppn1_positives'], tf.float32), name="ppn1_positives"))
                    tf.summary.scalar('ppn2_positives', tf.reduce_sum(tf.cast(self._predictions['ppn2_positives'], tf.float32), name="ppn2_positives"))

                    tf.summary.scalar('loss_ppn1_point', self._losses['loss_ppn1_point'])
                    tf.summary.scalar('loss_ppn1_class', self._losses['loss_ppn1_class'])
                    tf.summary.scalar('loss_ppn2_point', self._losses['loss_ppn2_point'])
                    tf.summary.scalar('loss_ppn2_class', self._losses['loss_ppn2_class'])
                    tf.summary.scalar('loss_ppn2_background', self._losses['loss_ppn2_background'])
                    tf.summary.scalar('loss_ppn2_track', self._losses['loss_ppn2_track'])
                    tf.summary.scalar('loss_ppn2_shower', self._losses['loss_ppn2_shower'])
                    tf.summary.scalar('accuracy_ppn1', self._predictions['accuracy_ppn1'])
                    tf.summary.scalar('accuracy_ppn2', self._predictions['accuracy_ppn2'])

                    with tf.variable_scope("optimizer"):
                        global_step = tf.Variable(0, trainable=False, name="global_step")
                        lr = tf.train.exponential_decay(self.lr, global_step, 10000, 0.95)
                        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
                        self.train_op = optimizer.minimize(total_loss, global_step=global_step)

                self.summary_op = tf.summary.merge_all()

                # Testing time
                # Turn predicted positions (float) into original image positions
                # Convert proposals2 ROI 1x1 coordinates to 64x64 F3 coordinates
                # then back to original image.
                # FIXME take top scores only? or leave it to the demo script
                with tf.variable_scope("final_proposals"):
                    im_proposals = tf.identity((proposals2 + self.dim2*rois)*self.dim1, name="im_proposals_raw")
                    im_labels = tf.argmax(scores2, axis=1, name="im_labels_raw")
                    im_scores = tf.gather_nd(scores2, tf.concat([tf.reshape(tf.range(0, tf.shape(im_labels)[0]), (-1, 1)), tf.cast(tf.reshape(im_labels, (-1, 1)), tf.int32)], axis=1), name="im_scores_raw")
                    # We have now num_roi proposals and corresponding labels in original image.
                    # Pixel NMS equivalent ?
                    keep = tf.where(tf.greater(im_scores, self.cfg.MIN_SCORE), name="keep_good_scores")
                    im_proposals = tf.gather_nd(im_proposals, keep, name="im_proposals")
                    im_labels = tf.gather_nd(im_labels, keep, name="im_labels")
                    im_scores = tf.gather_nd(im_scores, keep, name="im_scores")
                    self._predictions['im_proposals'] = im_proposals
                    self._predictions['im_labels'] = im_labels
                    self._predictions['im_scores'] = im_scores

    def build_ppn1(self, net2):
        # =====================================================
        # ---       Pixel Proposal Network 1                ---
        # =====================================================
        with tf.variable_scope("ppn1", reuse=self.reuse):
            # Define initializers
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

            # Step 0) Convolution
            # Shape of ppn1 = 1, 16, 16, 512
            ppn1 = self.conv(net2,
                              self.ppn1_channels, # RPN Channels = num_outputs
                              3, # RPN Kernels
                              weights_initializer=initializer,
                              trainable=self.is_training,
                              scope="ppn1_conv/3x3")
            # Step 1-a) PPN 2 pixel position predictions
            # Shape of ppn1_pixel_pred = 1, 16, 16, 2
            ppn1_pixel_pred = self.conv(ppn1, self.dim, 1,
                                        weights_initializer=initializer,
                                        trainable=self.is_training,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn1_pixel_pred')
            # Step 1-b) Generate 2 class scores (background vs signal)
            # Shape of ppn1_cls_score = 1, 16, 16, 2
            ppn1_cls_score = self.conv(ppn1, 2, 1,
                                        weights_initializer=initializer,
                                        trainable=self.is_training,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn1_cls_score')

            # Compute softmax
            # Shape of ppn1_cls_prob = 1, 16, 16, 2
            ppn1_cls_prob = tf.nn.softmax(ppn1_cls_score)

            # Step 3) Get a (meaningful) subset of rois and associated scores
            # Generate anchors = pixel centers of the last feature map.
            # Shape of anchors = 16*16, 2
            anchors = generate_anchors((self.N3,) * self.dim)
            assert anchors.get_shape().as_list() == [self.N3**self.dim, self.dim]

            # Derive predicted positions (poi) with scores (poi_scores) from prediction parameters
            # and anchors. Take the first R proposed pixels which contain an object.
            proposals, scores = predicted_pixels(ppn1_cls_prob, ppn1_pixel_pred, anchors, (self.N2,) * self.dim)
            rois, roi_scores = top_R_pixels(proposals, scores, R=20, threshold=self.ppn1_score_threshold)
            assert proposals.get_shape().as_list() == [self.N3**self.dim, self.dim]
            assert scores.get_shape().as_list() == [self.N3**self.dim, 1]
            #assert rois.get_shape().as_list() == [None, 2]
            #assert roi_scores.get_shape().as_list() == [None, 1]

            self._predictions['ppn1_pixel_pred'] = ppn1_pixel_pred # Pixel predictions
            self._predictions['ppn1_cls_score'] = ppn1_cls_score # Background vs signal scores
            self._predictions['ppn1_cls_prob'] = ppn1_cls_prob # After softmax
            self._predictions['ppn1_anchors'] = anchors
            self._predictions['ppn1_proposals'] = proposals
            self._predictions['ppn1_scores'] = scores

            #if self.is_training:
            # all outputs from 1x1 convolution are categorized into “positives” and “negatives”.
            # Positives = pixels which contain a ground-truth point
            # Negatives = other pixels
            classes_mask = compute_positives_ppn1(self.get_gt_pixels(), self.N3, self.dim1, self.dim2)
            assert classes_mask.get_shape().as_list() == [self.N3**self.dim, 1]
            # FIXME Use Kazu's pixel index to limit the number of gt points for
            # which we compute a distance from a unique proposed point per pixel.

            # For each pixel of the F5 features map get distance between proposed point
            # and the closest ground truth pixel
            # Don't forget to convert gt pixels coordinates to F5 coordinates
            closest_gt, closest_gt_distance, _ = assign_gt_pixels(self.gt_pixels_placeholder, proposals, self.dim1, self.dim2)
            assert closest_gt.get_shape().as_list() == [self.N3**self.dim]
            assert closest_gt_distance.get_shape().as_list() == [self.N3**self.dim, 1]
            #assert closest_gt_label.get_shape().as_list() == [256, 1]
            self._predictions['ppn1_closest_gt'] = closest_gt
            self._predictions['ppn1_closest_gt_distance'] = closest_gt_distance

            # Step 4) compute loss for PPN1
            # First is point loss: for positive pixels, distance from proposed pixel to closest ground truth pixel
            # FIXME Use smooth L1 for distance loss?
            loss_ppn1_point = tf.reduce_mean(tf.reduce_mean(tf.boolean_mask(closest_gt_distance, classes_mask)), name="loss_ppn1_point")
            #loss_ppn1_point = tf.reduce_mean(tf.reduce_mean(tf.exp(1.0 * tf.boolean_mask(closest_gt_distance, classes_mask))))
            # Use softmax_cross_entropy instead of sigmoid here
            #loss_ppn1_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(classes_mask, tf.float32), logits=scores))
            labels_ppn1 = tf.cast(tf.reshape(classes_mask, (-1,)), tf.int32, name="labels_ppn1")
            loss_ppn1_class = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn1,
                                                                                            logits=tf.reshape(ppn1_cls_score, (-1, 2)))), name="loss_ppn1_class")
            accuracy_ppn1 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(tf.reshape(ppn1_cls_prob, (-1, 2)), axis=1), tf.int32), labels_ppn1), tf.float32), name="accuracy_ppn1")

            self._predictions['ppn1_positives'] = classes_mask
            self._predictions['labels_ppn1'] = labels_ppn1
            self._losses['loss_ppn1_point'] =  loss_ppn1_point
            self._losses['loss_ppn1_class'] = loss_ppn1_class
            self._predictions['accuracy_ppn1'] = accuracy_ppn1

            return rois
        # --- END of Pixel Proposal Network 1 ---

    def build_ppn2(self, rpn_pooling, rois):
        # =====================================================
        # ---         Pixel Proposal Network 2              ---
        # =====================================================
        with tf.variable_scope("ppn2", reuse=self.reuse):
            # Define initializers
            initializer2=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            batch_size = tf.shape(rpn_pooling)[0] # should be number of rois x number of pixels per roi
            # Step 0) Convolution for PPN2 intermediate layer
            # Based on F3 feature map (ie after 3 max-pool layers in VGG)
            # Shape = nb_rois, 1, 1, 512
            ppn2 = self.conv(rpn_pooling,
                              self.ppn2_channels, # RPN Channels = num_outputs
                              3, # RPN Kernels FIXME change this to (1, 1)?
                              trainable=self.is_training,
                              weights_initializer=initializer2,
                              scope="ppn2_conv/3x3")
            # Step 1-a) PPN 2 pixel prediction parameters
            # Proposes pixel position (x, y) w.r.t. pixel center = anchor
            # Shape of ppn2_pixel_pred = nb_rois, 1, 1, 2
            ppn2_pixel_pred = self.conv(ppn2, self.dim, 1,
                                        trainable=self.is_training,
                                        weights_initializer=initializer2,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn2_pixel_pred')
            # Step 1-b) Generate class scores
            # Shape of ppn2_cls_score = nb_rois, 1, 1, num_classes
            ppn2_cls_score = self.conv(ppn2, self.num_classes, 1,
                                        trainable=self.is_training,
                                        weights_initializer=initializer2,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn2_cls_score')
            # Compute softmax
            ppn2_cls_prob = tf.nn.softmax(ppn2_cls_score)

            # Step 3) Get a (meaningful) subset of rois and associated scores
            # Anchors are defined as center of pixels
            # Shape [nb_rois * 1 * 1, 2]
            anchors2 = generate_anchors((1,)*self.dim, repeat=batch_size)
            assert anchors2.get_shape().as_list() == [None, self.dim]
            # Derive proposed points from delta predictions (ppn2_pixel_pred) w.r.t. pixels centers
            # Coordinates of proposals2 are in 1x1 ROI area
            # We have 1*1*num_roi proposals and corresponding scores
            proposals2, scores2 = predicted_pixels(ppn2_cls_prob, ppn2_pixel_pred, anchors2, (1,)*self.dim)
            assert proposals2.get_shape().as_list() == [None, self.dim]
            assert scores2.get_shape().as_list() == [None, self.num_classes-1]

            self._predictions['ppn2_pixel_pred'] = ppn2_pixel_pred
            self._predictions['ppn2_cls_score'] = ppn2_cls_score
            self._predictions['ppn2_cls_prob'] = ppn2_cls_prob
            self._predictions['ppn2_anchors'] = anchors2
            self._predictions['ppn2_proposals'] = proposals2
            self._predictions['ppn2_scores'] = scores2

            #if self.is_training:
            # Find closest ground truth pixel and its label
            # Option roi allows to convert gt_pixels_placeholder information to ROI 4x4 coordinates
            # closest_gt, closest_gt_distance, true_labels = assign_gt_pixels(self.gt_pixels_placeholder, proposals2, rois=rois, scores=ppn2_cls_score)
            closest_gt, closest_gt_distance, true_labels = assign_gt_pixels(self.gt_pixels_placeholder, proposals2, self.dim1, self.dim2, rois=rois)
            # assert closest_gt.get_shape().as_list() == [None]
            # assert closest_gt_distance.get_shape().as_list() == [None, 1]
            # assert true_labels.get_shape().as_list() == [None, 1]

            # Positives now = pixels within certain distance range from
            # the closest ground-truth point of the same class (track edge or shower start)
            positives = compute_positives_ppn2(closest_gt_distance, threshold=self.ppn2_distance_threshold)
            # assert positives.get_shape().as_list() == [None, 1]

            # Step 4) Loss
            # first is based on an absolute distance to the closest
            # ground-truth point where only positives count
            loss_ppn2_point = tf.reduce_mean(tf.reduce_mean(tf.boolean_mask(closest_gt_distance, positives)), name="loss_ppn2_point")
            # loss_ppn2_point = tf.reduce_mean(tf.reduce_mean(tf.exp(1.0 * tf.boolean_mask(closest_gt_distance, positives))))
            # second is a softmax class loss from both positives and negatives
            # the true label is defined by the closest ground truth point’s label
            # Negatives true labels should be background = 0
            labels_ppn2 = tf.cast(tf.reshape(tf.cast(positives, tf.float32)*true_labels, (-1,)), tf.int32, name="labels_ppn2")
            logits = tf.reshape(ppn2_cls_score, (-1, self.num_classes), name="ppn2_logits")

            if self.cfg.WEIGHT_LOSS:
                # FIXME class hardcoded?
                track_indices = tf.where(tf.equal(labels_ppn2, tf.constant(1)), name="track_indices")
                shower_indices = tf.where(tf.equal(labels_ppn2, tf.constant(2)), name="shower_indices")
                background_indices = tf.where(tf.equal(labels_ppn2, tf.constant(0)), name="background_indices")

                labels_ppn2_track = tf.gather(labels_ppn2, track_indices, name="labels_ppn2_track")
                labels_ppn2_shower = tf.gather(labels_ppn2, shower_indices, name="labels_ppn2_shower")
                labels_ppn2_background = tf.gather(labels_ppn2, background_indices, name="labels_ppn2_background")

                logits_track = tf.gather(logits, track_indices, name="logits_track")
                logits_shower = tf.gather(logits, shower_indices, name="logits_shower")
                logits_background = tf.gather(logits, background_indices, name="logits_background")

                loss_ppn2_background = tf.cond(tf.equal(tf.shape(background_indices)[0], tf.constant(0)), false_fn=lambda: tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn2_background, logits=logits_background))), true_fn=lambda: 0.0, name="loss_ppn2_background")
                loss_ppn2_track = tf.cond(tf.equal(tf.shape(track_indices)[0], tf.constant(0)), false_fn=lambda: tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn2_track, logits=logits_track))), true_fn=lambda: 0.0, name="loss_ppn2_track")
                loss_ppn2_shower = tf.cond(tf.equal(tf.shape(shower_indices)[0], tf.constant(0)), false_fn=lambda: tf.reduce_mean(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn2_shower, logits=logits_shower))), true_fn=lambda: 0.0, name="loss_ppn2_shower")

                gt_labels = tf.slice(self.gt_pixels_placeholder, [0, self.dim], [-1, 1], name="gt_labels")
                nb_tracks = tf.reduce_sum(tf.cast(tf.equal(gt_labels, 1), tf.float32), name="nb_tracks")
                nb_showers = tf.reduce_sum(tf.cast(tf.equal(gt_labels, 2), tf.float32), name="nb_showers")
                loss_ppn2_class = tf.identity(loss_ppn2_background + nb_tracks / (nb_tracks + nb_showers) * loss_ppn2_track + nb_showers / (nb_tracks + nb_showers) * loss_ppn2_shower, name="loss_ppn2_class")
            else:
                loss_ppn2_class = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn2, logits=logits)), name="loss_ppn2_class")

            accuracy_ppn2 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(tf.reshape(ppn2_cls_prob, (-1, self.num_classes)), axis=1), tf.int32), labels_ppn2), tf.float32))

            self._predictions['ppn2_true_labels'] = true_labels
            self._predictions['ppn2_positives'] = positives
            self._losses['loss_ppn2_point'] = loss_ppn2_point
            self._losses['loss_ppn2_class'] = loss_ppn2_class
            self._losses['loss_ppn2_background'] = loss_ppn2_background
            self._losses['loss_ppn2_track'] = loss_ppn2_track
            self._losses['loss_ppn2_shower'] = loss_ppn2_shower
            self._predictions['accuracy_ppn2'] = accuracy_ppn2
            self._predictions['ppn2_true_labels'] = true_labels
            self._predictions['ppn2_closest_gt_distance'] = closest_gt_distance
            self._predictions['ppn2_closest_gt'] = closest_gt

            return proposals2, scores2
            # --- END of Pixel Proposal Network 2 ---

    def get_gt_pixels(self):
        """
        Slice first 2 dimensions of gt_pixels_placeholder (coordinates only)
        We want it to be shape (None, 2)
        """
        # FIXME check that this returns the expected
        # return tf.squeeze(self.gt_pixels_placeholder, axis=[2])
        return tf.slice(self.gt_pixels_placeholder, [0, 0], [-1, self.dim], name="gt_pixels_coord")

if __name__ == "__main__":
    net = PPN()
    net.create_architecture()
