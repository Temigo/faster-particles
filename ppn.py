# *-* encoding: utf-8 *-*
# Pixel Proposal Network
# Draft implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import sys, os

from ppn_utils import include_gt_pixels, compute_positives_ppn2, \
    compute_positives_ppn1, assign_gt_pixels, generate_anchors, \
    predicted_pixels, top_R_pixels, build_vgg

class PPN(object):

    def __init__(self, R=20, num_classes=3, N=512, build_base_net=build_vgg):
        """
        Allow for easy implementation of different base network architecture:
        build_base_net should take as inputs
        (image_placeholder, is_training=True, reuse=False)
        and return (F3, F5) conv layers
        """
        # Global parameters
        self.R = R
        self.num_classes = num_classes # (B)ackground, (T)rack edge, (S)hower start, (S+T)
        self.N = N
        self.ppn1_score_threshold = 0.5
        self.ppn2_distance_threshold = 5
        self.lr = 0.001 # Learning rate
        self.lambda_ppn1 = 0.5 # Balance loss between class and distance in ppn1
        self.lambda_ppn2 = 0.5 # Balance loss between class and distance in ppn2
        self.lambda_ppn = 0.5 # Balance loss between ppn1 and ppn2
        self._predictions = {}
        self._losses = {}
        self.build_base_net = build_base_net

    def test_image(self, sess, blob):
        feed_dict = { self.image_placeholder: blob['data'] }
        im_proposals, im_labels, im_scores, ppn1_proposals, \
        rois, ppn2_proposals = sess.run([
            self._predictions['im_proposals'],
            self._predictions['im_labels'],
            self._predictions['im_scores'],
            self._predictions['ppn1_proposals'],
            self._predictions['rois'],
            self._predictions['ppn2_proposals']
            ], feed_dict=feed_dict)
        return im_proposals, im_labels, im_scores, ppn1_proposals, rois, ppn2_proposals

    def get_summary(self, sess, blobs):
        feed_dict = { self.image_placeholder: blobs['data'], self.gt_pixels_placeholder: blobs['gt_pixels'] }
        summary = sess.run(self.summary_op, feed_dict=feed_dict)
        return summary

    # FIXME train_op argument useless?
    def train_step(self, sess, blobs, train_op):
        feed_dict = { self.image_placeholder: blobs['data'], self.gt_pixels_placeholder: blobs['gt_pixels'] }
        _, total_loss = sess.run([self.train_op, self._losses['total_loss']], feed_dict=feed_dict)
        return total_loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = { self.image_placeholder: blobs['data'], self.gt_pixels_placeholder: blobs['gt_pixels'] }
        _, ppn1_pixel_pred, ppn1_cls_prob, ppn1_anchors, ppn1_proposals, \
        ppn1_scores, labels_ppn1, rois, ppn2_anchors, ppn2_proposals, ppn2_positives, \
        ppn2_scores, ppn2_closest_gt_distance, ppn2_true_labels, ppn2_cls_score, \
        loss_ppn2_point, summary = sess.run([
                            self.train_op,
                            self._predictions['ppn1_pixel_pred'],
                            self._predictions['ppn1_cls_prob'],
                            self._predictions['ppn1_anchors'],
                            self._predictions['ppn1_proposals'],
                            self._predictions['ppn1_scores'],
                            self._predictions['labels_ppn1'],
                            self._predictions['rois'],
                            self._predictions['ppn2_anchors'],
                            self._predictions['ppn2_proposals'],
                            self._predictions['ppn2_positives'],
                            self._predictions['ppn2_scores'],
                            self._predictions['ppn2_closest_gt_distance'],
                            self._predictions['ppn2_true_labels'],
                            self._predictions['ppn2_cls_score'],
                            self._losses['loss_ppn2_point'],
                            self.summary_op
                            ], feed_dict=feed_dict)

        #print("ppn1_pixel_pred : ", ppn1_pixel_pred.shape, ppn1_pixel_pred[0][0])
        #print("ppn1_cls_prob : ", ppn1_cls_prob.shape, ppn1_cls_prob[0][0])
        #print("ppn1_anchors : ", ppn1_anchors.shape, ppn1_anchors[0:10])
        #print("ppn1_proposals : ", ppn1_proposals.shape, ppn1_proposals[0:10])
        #print("ppn1_scores : ", ppn1_scores.shape, ppn1_scores[0:10])
        #print("labels ppn1 : ", labels_ppn1.shape, labels_ppn1)
        #print("ppn1 rois: ", rois.shape, rois[0:10])
        print("ppn2_anchors: ", ppn2_anchors.shape, ppn2_anchors[0:10])
        print("ppn2_proposals : ", ppn2_proposals.shape, ppn2_proposals[0:10])
        print("ppn2_positives: ", ppn2_positives.shape, ppn2_positives[0:10])
        print("ppn2_scores: ", ppn2_scores.shape, ppn2_scores[0:10])
        print("ppn2_closest_gt_distance: ", ppn2_closest_gt_distance.shape, ppn2_closest_gt_distance[0:10])
        print("ppn2_true_labels: ", ppn2_true_labels.shape, ppn2_true_labels[0:10])
        #print("ppn2_true_labels_both: ", ppn2_true_labels_both.shape, ppn2_true_labels_both[0:10])
        #print("difference=", np.sum(np.abs(ppn2_true_labels - ppn2_true_labels_both)))
        print("ppn2_cls_scores: ", ppn2_cls_score.shape, ppn2_cls_score[0:10])
        print("#positives: ", np.sum(ppn2_positives))
        loss_ppn2_point_np = np.mean(np.mean(ppn2_closest_gt_distance[ppn2_positives]))
        print(loss_ppn2_point, loss_ppn2_point_np)
        #assert np.isclose(loss_ppn2_point, loss_ppn2_point_np)


        return summary, ppn1_proposals, labels_ppn1, rois, ppn2_proposals, ppn2_positives

    # def get_variables_to_restore(self, variables, var_keep_dic)
    # def get_summary(self, sess, blobs_val)
    # def fix_variables(self, sess, self.pretrained_model)

    def create_architecture(self, is_training=True, reuse=False):
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope("input", reuse=self.reuse):
            # Define placeholders
            #with tf.variable_scope("placeholders", reuse=self.reuse):
            # FIXME Assuming batch size of 1 currently
            self.image_placeholder       = tf.placeholder(name="image", shape=(1, 512, 512, 3), dtype=tf.float32)
            # Shape of gt_pixels_placeholder = nb_gt_pixels, 2 coordinates + 1 class label in [0, num_classes)
            self.gt_pixels_placeholder   = tf.placeholder(name="gt_pixels", shape=(None, 3), dtype=tf.float32)

        # Define network regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        biases_regularizer = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            trainable=self.is_training,
                            weights_regularizer=weights_regularizer,
                            biases_regularizer=biases_regularizer,
                            biases_initializer=tf.constant_initializer(0.0)):        
            # Returns F3 and F5 feature maps
            net, net2 = self.build_base_net(self.image_placeholder, is_training=self.is_training, reuse=self.reuse)
            with tf.variable_scope("ppn", reuse=self.reuse):
                # Build PPN1
                rois = self.build_ppn1(net2)

                if self.is_training:
                    # During training time, check if all ground truth pixels are covered by ROIs
                    # If not, add relevant ROIs on F3
                    # TODO Algorithm should not place 4x4 exactly centered around ground-truth point,
                    # but instead allow random variation
                    rois = include_gt_pixels(rois, self.get_gt_pixels())
                    assert rois.get_shape().as_list() == [None, 2]

                self._predictions['rois'] = rois

                # Pool to Pixels of Interest of intermediate layer
                # FIXME How do we want to do the ROI pooling?
                # Shape of rpn_pooling = nb_rois, 4, 4, 256
                rpn_pooling = self.crop_pool_layer_2d(net, rois)
                assert rpn_pooling.get_shape().as_list() == [None, 1, 1, 256]

                proposals2, scores2 = self.build_ppn2(rpn_pooling, rois)

                if self.is_training:
                    tf.summary.scalar('ppn1_positives', tf.reduce_sum(tf.cast(self._predictions['ppn1_positives'], tf.float32)))
                    tf.summary.scalar('ppn2_positives', tf.reduce_sum(tf.cast(self._predictions['ppn2_positives'], tf.float32)))
                    # FIXME How to combine losses
                    total_loss = self.lambda_ppn * (self.lambda_ppn1 * self._losses['loss_ppn1_point'] \
                                + (1.0 - self.lambda_ppn1) * self._losses['loss_ppn1_class']) \
                                + (1.0 - self.lambda_ppn) * (self.lambda_ppn2 * self._losses['loss_ppn2_point'] \
                                + (1.0 - self.lambda_ppn2) * self._losses['loss_ppn2_class'])
                    self._losses['total_loss'] = total_loss
                    tf.summary.scalar('loss', total_loss)
                    tf.summary.scalar('loss_ppn1_point', self._losses['loss_ppn1_point'])
                    tf.summary.scalar('loss_ppn1_class', self._losses['loss_ppn1_class'])
                    tf.summary.scalar('loss_ppn2_point', self._losses['loss_ppn2_point'])
                    tf.summary.scalar('loss_ppn2_class', self._losses['loss_ppn2_class'])
                    tf.summary.scalar('accuracy_ppn1', self._predictions['accuracy_ppn1'])
                    tf.summary.scalar('accuracy_ppn2', self._predictions['accuracy_ppn2'])
                    self.summary_op = tf.summary.merge_all()

                    global_step = tf.Variable(0, trainable=False)
                    lr = tf.train.exponential_decay(self.lr, global_step, 1000, 0.95)
                    optimizer = tf.train.AdamOptimizer(lr)
                    self.train_op = optimizer.minimize(total_loss, global_step=global_step)

                # Testing time
                # Turn predicted positions (float) into original image positions
                # Convert proposals2 ROI 1x1 coordinates to 64x64 F3 coordinates
                # then back to original image.
                # FIXME take top scores only? or leave it to the demo script
                im_proposals = (proposals2 + 4*rois)*8.0
                im_labels = tf.argmax(scores2, axis=1)
                im_scores = tf.gather(scores2, im_labels)
                self._predictions['im_proposals'] = im_proposals
                self._predictions['im_labels'] = im_labels
                self._predictions['im_scores'] = im_scores
                # We have now num_roi proposals and corresponding labels in original image.
                # Pixel NMS equivalent ?

    def build_ppn1(self, net2):
        # =====================================================
        # ---       Pixel Proposal Network 1                ---
        # =====================================================
        with tf.variable_scope("ppn1", reuse=self.reuse):
            # Define initializers
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

            # Step 0) Convolution for RPN/Detection shared layer
            # Shape of rpn = 1, 16, 16, 512
            ppn1 = slim.conv2d(net2,
                              512, # RPN Channels = num_outputs
                              (3, 3), # RPN Kernels
                              weights_initializer=initializer,
                              trainable=self.is_training,
                              scope="ppn1_conv/3x3")
            # Step 1-a) PPN 2 pixel position predictions
            # Shape of rpn_bbox_pred = 1, 16, 16, 2
            ppn1_pixel_pred = slim.conv2d(ppn1, 2, [1, 1],
                                        weights_initializer=initializer,
                                        trainable=self.is_training,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn1_pixel_pred')
            # Step 1-b) Generate 2 class scores (background vs signal)
            # Shape of rpn_cls_score = 1, 16, 16, 2
            # FIXME use sigmoid instead of softmax?
            ppn1_cls_score = slim.conv2d(ppn1, 2, [1, 1],
                                        weights_initializer=initializer,
                                        trainable=self.is_training,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn1_cls_score')

            # Compute softmax
            # Shape of rpn_cls_prob = 1, 16, 16, 2
            ppn1_cls_prob = tf.nn.softmax(ppn1_cls_score)

            # Step 3) Get a (meaningful) subset of rois and associated scores
            # Generate anchors = pixel centers of the last feature map.
            # Shape of anchors = 16*16, 2
            anchors = generate_anchors(width=16, height=16) # FIXME express width and height better
            assert anchors.get_shape().as_list() == [256, 2]

            # Derive predicted positions (poi) with scores (poi_scores) from prediction parameters
            # and anchors. Take the first R proposed pixels which contain an object.
            proposals, scores = predicted_pixels(ppn1_cls_prob, ppn1_pixel_pred, anchors, (16, 16)) # FIXME hardcoded
            rois, roi_scores = top_R_pixels(proposals, scores, R=20, threshold=self.ppn1_score_threshold)
            assert proposals.get_shape().as_list() == [256, 2]
            assert scores.get_shape().as_list() == [256, 1]
            assert rois.get_shape().as_list() == [None, 2]
            assert roi_scores.get_shape().as_list() == [None, 1]

            self._predictions['ppn1_pixel_pred'] = ppn1_pixel_pred # Pixel predictions
            self._predictions['ppn1_cls_score'] = ppn1_cls_score # Background vs signal scores
            self._predictions['ppn1_cls_prob'] = ppn1_cls_prob # After softmax
            self._predictions['ppn1_anchors'] = anchors
            self._predictions['ppn1_proposals'] = proposals
            self._predictions['ppn1_scores'] = scores

            if self.is_training:
                # all outputs from 1x1 convolution are categorized into “positives” and “negatives”.
                # Positives = pixels which contain a ground-truth point
                # Negatives = other pixels
                classes_mask = compute_positives_ppn1(self.get_gt_pixels())
                assert classes_mask.get_shape().as_list() == [256, 1]
                # FIXME Use Kazu's pixel index to limit the number of gt points for
                # which we compute a distance from a unique proposed point per pixel.

                # For each pixel of the F5 features map get distance between proposed point
                # and the closest ground truth pixel
                # Don't forget to convert gt pixels coordinates to F5 coordinates
                closest_gt, closest_gt_distance, _ = assign_gt_pixels(self.gt_pixels_placeholder, proposals)
                assert closest_gt.get_shape().as_list() == [256]
                assert closest_gt_distance.get_shape().as_list() == [256, 1]
                #assert closest_gt_label.get_shape().as_list() == [256, 1]

                # Step 4) compute loss for PPN1
                # First is point loss: for positive pixels, distance from proposed pixel to closest ground truth pixel
                # FIXME Use smooth L1 for distance loss?
                loss_ppn1_point = tf.reduce_mean(tf.reduce_mean(tf.boolean_mask(closest_gt_distance, classes_mask)))
                #loss_ppn1_point = tf.reduce_mean(tf.reduce_mean(tf.exp(1.0 * tf.boolean_mask(closest_gt_distance, classes_mask))))
                # Use softmax_cross_entropy instead of sigmoid here
                #loss_ppn1_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(classes_mask, tf.float32), logits=scores))
                labels_ppn1 = tf.cast(tf.reshape(classes_mask, (-1,)), tf.int32)
                loss_ppn1_class = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn1,
                                                                                                logits=tf.reshape(ppn1_cls_score, (-1, 2)))))
                accuracy_ppn1 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(tf.reshape(ppn1_cls_prob, (-1, 2)), axis=1), tf.int32), labels_ppn1), tf.float32))

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
            ppn2 = slim.conv2d(rpn_pooling,
                              512, # RPN Channels = num_outputs
                              (3, 3), # RPN Kernels FIXME change this to (1, 1)?
                              trainable=self.is_training,
                              weights_initializer=initializer2,
                              scope="ppn2_conv/3x3")
            # Step 1-a) PPN 2 pixel prediction parameters
            # Proposes pixel position (x, y) w.r.t. pixel center = anchor
            # Shape of rpn_bbox_pred2 = nb_rois, 1, 1, 2
            ppn2_pixel_pred = slim.conv2d(ppn2, 2, [1, 1],
                                        trainable=self.is_training,
                                        weights_initializer=initializer2,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn2_pixel_pred2')
            # Step 1-b) Generate class scores
            # Shape of rpn_cls_score2 = nb_rois, 1, 1, num_classes
            ppn2_cls_score = slim.conv2d(ppn2, self.num_classes, [1, 1],
                                        trainable=self.is_training,
                                        weights_initializer=initializer2,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='ppn2_cls_score')
            # Compute softmax
            ppn2_cls_prob = tf.nn.softmax(ppn2_cls_score) # FIXME might need a reshape here

            # Step 3) Get a (meaningful) subset of rois and associated scores
            # Anchors are defined as center of pixels
            # Shape [nb_rois * 4 * 4 , 2]
            anchors2 = generate_anchors(width=1, height=1, repeat=batch_size) # FIXME express width and height better
            assert anchors2.get_shape().as_list() == [None, 2]
            # Derive proposed points from delta predictions (rpn_bbox_pred2) w.r.t. pixels centers
            # Coordinates of proposals2 are in 1x1 ROI area
            # We have 1*1*num_roi proposals and corresponding scores
            proposals2, scores2 = predicted_pixels(ppn2_cls_prob, ppn2_pixel_pred, anchors2, (1, 1), classes=True)
            assert proposals2.get_shape().as_list() == [None, 2]
            assert scores2.get_shape().as_list() == [None, self.num_classes-1]

            self._predictions['ppn2_pixel_pred'] = ppn2_pixel_pred
            self._predictions['ppn2_cls_score'] = ppn2_cls_score
            self._predictions['ppn2_cls_prob'] = ppn2_cls_prob
            self._predictions['ppn2_anchors'] = anchors2
            self._predictions['ppn2_proposals'] = proposals2
            self._predictions['ppn2_scores'] = scores2

            if self.is_training:
                # Find closest ground truth pixel and its label
                # Option roi allows to convert gt_pixels_placeholder information to ROI 4x4 coordinates
                # closest_gt, closest_gt_distance, true_labels = assign_gt_pixels(self.gt_pixels_placeholder, proposals2, rois=rois, scores=ppn2_cls_score)
                closest_gt, closest_gt_distance, true_labels = assign_gt_pixels(self.gt_pixels_placeholder, proposals2, ppn2=True)
                assert closest_gt.get_shape().as_list() == [None]
                assert closest_gt_distance.get_shape().as_list() == [None, 1]
                assert true_labels.get_shape().as_list() == [None, 1]

                # Positives now = pixels within certain distance range from
                # the closest ground-truth point of the same class (track edge or shower start)
                positives = compute_positives_ppn2(scores2, closest_gt_distance, true_labels, threshold=self.ppn2_distance_threshold)
                assert positives.get_shape().as_list() == [None, 1]

                # Step 4) Loss
                # first is based on an absolute distance to the closest
                # ground-truth point where only positives count
                loss_ppn2_point = tf.reduce_mean(tf.reduce_mean(tf.boolean_mask(closest_gt_distance, positives)))
                # loss_ppn2_point = tf.reduce_mean(tf.reduce_mean(tf.exp(1.0 * tf.boolean_mask(closest_gt_distance, positives))))
                # second is a softmax class loss from both positives and negatives
                # the true label is defined by the closest ground truth point’s label
                #_, _, true_labels_both = assign_gt_pixels(self.gt_pixels_placeholder, proposals2)
                labels_ppn2 = tf.cast(tf.reshape(true_labels, (-1,)), tf.int32)
                loss_ppn2_class = tf.reduce_mean(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_ppn2,
                                                                                 logits=tf.reshape(ppn2_cls_score, (-1, self.num_classes)))))

                accuracy_ppn2 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(tf.reshape(ppn2_cls_prob, (-1, self.num_classes)), axis=1), tf.int32), labels_ppn2), tf.float32))

                self._predictions['ppn2_positives'] = positives
                self._losses['loss_ppn2_point'] = loss_ppn2_point
                self._losses['loss_ppn2_class'] = loss_ppn2_class
                self._predictions['accuracy_ppn2'] = accuracy_ppn2
                self._predictions['ppn2_true_labels'] = true_labels
                self._predictions['ppn2_closest_gt_distance'] = closest_gt_distance

            return proposals2, scores2
            # --- END of Pixel Proposal Network 2 ---

    def get_gt_pixels(self):
        """
        Slice first 2 dimensions of gt_pixels_placeholder (coordinates only)
        We want it to be shape (None, 2)
        """
        # FIXME check that this returns the expected
        # return tf.squeeze(self.gt_pixels_placeholder, axis=[2])
        return tf.slice(self.gt_pixels_placeholder, [0, 0], [-1, 2])

    def crop_pool_layer_2d(self, net, rois, R=20):
        """
        Crop and pool intermediate F3 layer.
        Net.shape = [1, 64, 64, 256]
        Rois.shape = [None, 2] # Could be less than R, assumes coordinates on F5
        Also assumes ROIs are 1x1 pixels on F3
        """
        with tf.variable_scope("crop_pool_layer"):
            assert net.get_shape().as_list() == [1, 64, 64, 256]
            assert rois.get_shape().as_list() == [None, 2]
            # Convert rois from F5 coordinates to F3 coordinates (x4)
            rois = (rois*4.0) # FIXME hardcoded
            # Shape of boxes = [num_boxes, 4]
            # boxes[i] is specified in normalized coordinates [y1, x1, y2, x2]
            # with y1 < y2 ideally
            boxes = tf.concat([rois, rois+1], axis=1)
            # then to normalized coordinates in [0, 1] of F3 feature map
            boxes = boxes / 64.0 # FIXME hardcoded
            assert boxes.get_shape().as_list() == [None, 4]

            # Shape of box_ind = [num_boxes] with values in [0, batch_size)
            # FIXME allow batch size > 1
            box_ind = tf.fill((tf.shape(rois)[0],), 0)
            # 1-D tensor of 2 elements = [crop_height, crop_width]
            # All cropped image patches are resized to this size
            # We want size 1x1 after max_pool2d
            crop_size = tf.constant([1*2, 1*2])
            crops = tf.image.crop_and_resize(net, boxes, box_ind, crop_size, name="crops1")
            # crops is a 4-D tensor of shape [num_boxes, crop_height, crop_width, depth]
            return slim.max_pool2d(crops, [2, 2], padding='SAME')

if __name__ == "__main__":
    net = PPN()
    net.create_architecture()
    # Dummy 4x4 image
    #dummy_rpn_cls_prob = np.ndarray([[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]])
    #dummy_rpn_bbox_pred = np.ndarray([])
    #dummy_anchors =
    #dummy_input_shape =
    #proposal_layer_2d(dummy_rpn_cls_prob, dummy_rpn_bbox_pred, dummy_anchors, dummy_input_shape)

    """image = tf.placeholder(tf.float32,[1,512,512,3])
    net.set_input_shape(image)
    # Create a session
    sess = tf.InteractiveSession()
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    #ret = sess.run(net._anchors,feed_dict={})
    #print('{:s}'.format(ret))"""
