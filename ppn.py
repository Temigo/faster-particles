# *-* encoding: utf-8 *-*
# Pixel Proposal Network
# Draft implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class PPN(object):

    def __init__(self, R=20, num_classes=3, N=512):
        # Global parameters
        self.R = R
        self.is_training = True
        self.num_classes = num_classes # background, track edge, shower start
        self.N = N

    def create_architecture(self):
        # Define placeholders
        self.image_placeholder       = tf.placeholder(name="image", shape=(1, 512, 512, 3), dtype=tf.float32)
        self.gt_pixels_placeholder   = tf.placeholder(name="gt_pixels", shape=(None, 2, 1), dtype=tf.float32)
        self.input_shape_placeholder = tf.placeholder(name="input_shape", shape=(4,), dtype=tf.int32)

        # Define network
        with tf.variable_scope("vgg_16"):
            # VGG16 net
            net = slim.repeat(self.image_placeholder, 2, slim.conv2d, 64, [3, 3],
                              trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                            trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                            trainable=self.is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net2 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                            trainable=self.is_training, scope='conv4')
            net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool4')
            net2 = slim.repeat(net2, 3, slim.conv2d, 512, [3, 3],
                            trainable=self.is_training, scope='conv5')
            net2 = slim.max_pool2d(net2, [2, 2], padding='SAME', scope='pool5')
            # After 5 times (2, 2) pooling, if input image is 512x512
            # the feature map should be spatial dimensions 16x16.

            # --- Pixel Proposal Network 1 ---
        with tf.variable_scope("ppn1"):
            # Define initializers
            rcnn_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Step 0) Convolution for RPN/Detection shared layer
            # Shape of rpn = 1, 16, 16, 512
            rpn = slim.conv2d(net2,
                              512, # RPN Channels = num_outputs
                              (3, 3), # RPN Kernels : (3, 3)
                              trainable=True,
                              weights_initializer=rcnn_initializer,
                              scope="rpn_conv/3x3")
            print("rpn shape:", rpn.shape)
            # Step 1-a) PPN 2 pixel position predictions
            # Shape of rpn_bbox_pred = 1, 16, 16, 2
            rpn_bbox_pred = slim.conv2d(rpn, 2, [1, 1],
                                        trainable=True,
                                        weights_initializer=rcnn_initializer,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='rpn_bbox_pred')
            print("rpn_bbox_pred shape:", rpn_bbox_pred.shape)
            # Step 1-b) Generate 2 class scores (background vs signal)
            # Shape of rpn_cls_score = 1, 16, 16, 2
            rpn_cls_score = slim.conv2d(rpn, 2, [1, 1],
                                        trainable=True,
                                        weights_initializer=rcnn_initializer,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='rpn_cls_score')
            print("rpn_cls_score shape:", rpn_cls_score.shape)

            # Compute softmax
            # Shape of rpn_cls_prob = 1, 16, 16, 2
            rpn_cls_prob = tf.nn.softmax(rpn_cls_score) # FIXME might need a reshape here
            print("rpn_cls_prob shape:", rpn_cls_prob.shape)

            # Step 3) Get a (meaningful) subset of rois and associated scores
            # Generate anchors = pixel centers of the last feature map.
            # Shape of anchors = 16*16, 2
            anchors = self.generate_anchors(width=16, height=16) # FIXME express width and height better

            # Derive predicted positions (poi) with scores (poi_scores) from prediction parameters
            # and anchors. Take the first R proposed pixels which contain an object.
            proposals, scores = self.predicted_pixels(rpn_cls_prob, rpn_bbox_pred, anchors)
            rois, roi_scores = self.top_R_pixels(proposals, scores, R=20)
            print("proposals shape=", proposals.shape)
            print("scores shape=", scores.shape)
            # Positives = pixels which contain a ground-truth point
            # TODO intersection with proposals
            classes = self.compute_positives_ppn1()
            closest_gt, closest_gt_distance, _ = self.assign_gt_pixels(proposals)

            # Step 4) compute loss for PPN1
            classes_mask = tf.cast(classes, tf.bool) # Turn classes into a mask
            # First is point loss: for positive pixels, distance from proposed pixel to closest ground truth pixel
            loss_ppn1_point = tf.reduce_mean(tf.boolean_mask(closest_gt_distance, classes_mask))
            # FIXME do softmax_cross_entropy instead of sigmoid here
            loss_ppn1_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(classes, tf.float32), logits=scores))

            # --- END of Pixel Proposal Network 1 ---

            # TODO make sure all gt pixels are covered by ROIs

            # Pool to Pixels of Interest of intermediate layer
            # FIXME How do we want to do the ROI pooling?
            # Shape of rpn_pooling = nb_rois, 64, 64, 256
            rpn_pooling = self.crop_pool_layer_2d(net, rois)
            print("rpn_pooling shape=", rpn_pooling.shape)

            # --- Pixel Proposal Network 2 ---
        with tf.variable_scope("ppn2"):
            # Define initializers
            rcnn_initializer2=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer2 = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox2 = tf.random_normal_initializer(mean=0.0, stddev=0.001)

            # Step 0) Convolution for RPN/Detection shared layer
            # Shape of rpn2 =
            rpn2 = slim.conv2d(rpn_pooling,
                              512, # RPN Channels = num_outputs
                              (3, 3), # RPN Kernels : (3, 3)
                              trainable=True,
                              weights_initializer=rcnn_initializer2,
                              scope="rpn_conv2/3x3")
            print("\nrpn2 shape=", rpn2.shape)
            # Step 1-a) PPN 2 pixel prediction parameters
            # Proposes pixel position (x, y) w.r.t. pixel center = anchor
            # Shape of rpn_bbox_pred2 = batch_size, 64, 64, 2
            rpn_bbox_pred2 = slim.conv2d(rpn2, 2, [1, 1],
                                        trainable=True,
                                        weights_initializer=rcnn_initializer2,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='rpn_bbox_pred2')
            print("rpn_bbox_pred2 shape=", rpn_bbox_pred2.shape)
            # Step 1-b) Generate class scores
            # Shape of rpn_cls_score2 = batch_size, 64, 64, num_classes
            rpn_cls_score2 = slim.conv2d(rpn2, self.num_classes, [1, 1],
                                        trainable=True,
                                        weights_initializer=rcnn_initializer2,
                                        padding='VALID',
                                        activation_fn=None,
                                        scope='rpn_cls_score2')
            print("rpn_cls_score2 shape=", rpn_cls_score2.shape)
            # Compute softmax
            rpn_cls_prob2 = tf.nn.softmax(rpn_cls_score2) # FIXME might need a reshape here
            print("rpn_cls_prob2 shape=", rpn_cls_prob2.shape)

            # Step 3) Get a (meaningful) subset of rois and associated scores
            anchors2 = self.generate_anchors(width=64, height=64) # FIXME express width and height better
            proposals2, scores2 = self.predicted_pixels(rpn_cls_prob2, rpn_bbox_pred2, anchors2, classes=True)
            print("proposals shape=", proposals2.shape)
            print("scores shape=", scores2.shape)
            closest_gt, closest_gt_distance, true_labels = self.assign_gt_pixels(proposals2)
            # Positives now = pixels within certain distance range from
            # the closest ground-truth point of the same class (track edge or shower start)
            positives = self.compute_positives_ppn2(scores2, closest_gt_distance, true_labels, threshold=2)
            print("positives shape=", positives.shape)

            # Step 4) Loss
            # first is based on an absolute distance to the closest
            # ground-truth point where only positives count
            loss_ppn2_point = tf.reduce_sum(tf.boolean_mask(closest_gt_distance, positives))
            # second is a softmax class loss from both positives and negatives
            # for positives, the true label is defined by the closest point’s label
            loss_ppn2_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(true_labels, (-1,)), tf.int32),
                                                                             logits=tf.reshape(rpn_cls_score2, (-1, self.num_classes)))

            # --- END of Pixel Proposal Network 2 ---

            # Pool to Pixels of Interest
            #rpn_pooling2 = crop_pool_layer_2d(net, rois2, "rpn_pooling") # FIXME net?

            # --- Pixel classification on 64x64 layer ---
            #net_flat = slim.flatten(rpn_pooling2, scope='flatten')
            #fc6 = slim.fully_connected(net_flat, 4096, scope='fc6')
            #if is_training:
            #    fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
            #                        scope='dropout6')
            #rcnn_input = slim.fully_connected(fc6, 4096, scope='fc7')
            #if is_training:
            #    rcnn_input = slim.dropout(rcnn_input, keep_prob=0.5, is_training=True,
            #                        scope='dropout7')
            #cls_prob, bbox_pred = region_classification_2d(rcnn_input, trainable,
            #                                                     initializer, initializer_bbox)

            # Turn predicted positions (float) into original image positions

    def compute_positives_ppn2(self, scores, closest_gt_distance, true_labels, threshold=2):
        """
        closest_gt_distance shape = (A*N*N, 1)
        true_labels shape = (A*N*N, 1)
        scores shape = (A*N*N, num_classes)
        Return boolean mask for positives among proposals.
        Positives are those within certain distance range from the
        closest ground-truth point of the same class
        """
        predicted_labels = tf.reshape(tf.argmax(scores, axis=1, output_type=tf.int32), true_labels.shape)
        print("predicted labels shape=", predicted_labels.shape)
        true_labels = tf.cast(true_labels, tf.int32)
        mask = tf.where(tf.greater(closest_gt_distance, threshold), tf.constant(value=False, shape=closest_gt_distance.shape), tf.constant(value=True, shape=closest_gt_distance.shape))
        mask = tf.where(tf.equal(true_labels, predicted_labels), mask, tf.constant(value=False, shape=closest_gt_distance.shape))
        return mask

    def assign_gt_pixels(self, proposals):
        """
        Proposals shape: [A*N*N, 2] (N=16 or 64)
        gt_pixels_placeholder is shape [None, 2, 1]
        Classes shape: [A*N*N, 1]
        Returns closest ground truth pixels for all pixels and corresponding distance
        """
        # Slice first 2 dimensions of gt_pixels_placeholder
        # We want it to be shape (None, 2)
        gt_pixels = tf.squeeze(self.gt_pixels_placeholder, axis=[2])
        # Tile to have shape (16*16, None, 2)
        gt_pixels = tf.expand_dims(gt_pixels, axis=0)
        all_gt_pixels = tf.tile(gt_pixels, (proposals.get_shape().as_list()[0], 1, 1)) # FIXME don't hardcode
        #all_gt_pixels = tf.transpose(all_gt_pixels, perm=[0, 2, 1])
        proposals = tf.expand_dims(proposals, axis=1)
        print("all_gt_pixels shape=", all_gt_pixels.shape)
        print("intermediate shape=", tf.reduce_sum(tf.pow(proposals - all_gt_pixels, 2), axis=2, keep_dims=True).shape)
        distances = tf.sqrt(tf.reduce_sum(tf.pow(proposals - all_gt_pixels, 2), axis=2))
        # distances.shape = [16*16, None, 1]
        print("distances shape=", distances.shape)
        # closest_gt.shape = [16*16,]
        # closest_gt[i] = indice of closest gt in gt_pixels_placeholder
        closest_gt = tf.argmin(distances, axis=1)
        print("closest_gt shape=", closest_gt.shape)
        closest_gt_distance = tf.reduce_min(distances, axis=1, keep_dims=True)
        print("closest_gt_distance shape=", closest_gt_distance.shape)
        #print("squeezed gt_pixels_placeholder shape=", tf.squeeze(tf.slice(gt_pixels_placeholder, [0,0,0], [-1,1,-1]), axis=1).shape)
        closest_gt_label = tf.nn.embedding_lookup(tf.squeeze(tf.slice(self.gt_pixels_placeholder, [0,0,0], [-1,1,-1]), axis=1), closest_gt)
        print("closest_gt_label shape=", closest_gt_label.shape)
        return closest_gt, closest_gt_distance, closest_gt_label

    def compute_positives_ppn1(self):
        """
        Returns a mask corresponding to proposals shape = [N*N, 2]
        Positive = 1 = contains a ground truth pixel
        gt_pixels_placeholder is shape [None, 2, 1]
        Returns classes with shape (16*16,1)
        """
        classes = tf.zeros(shape=(16, 16)) # FIXME don't hardcode 16
        # Slice first 2 dimensions of gt_pixels_placeholder
        #gt_pixels = tf.slice(gt_pixels_placeholder, [0, 0, 0], [-1, -1, 0])
        gt_pixels = tf.squeeze(self.gt_pixels_placeholder, axis=[2])
        # Convert to F5 coordinates (16x16)
        # Shape = None, 2
        gt_pixels = tf.cast(tf.floor(gt_pixels / 32.0), tf.int32)
        print("gt_pixels shape=", gt_pixels.shape)
        # Assign positive pixels based on gt_pixels
        #classes = classes + tf.scatter_nd(gt_pixels, tf.constant(value=1.0, shape=tf.shape(gt_pixels)[0]), classes.shape)
        classes = classes + tf.scatter_nd(gt_pixels, tf.fill((tf.shape(gt_pixels)[0],), 1.0), classes.shape)
        classes = tf.cast(tf.reshape(classes, shape=(-1, 1)), tf.int32)
        print("classes shape=", classes.shape)
        return classes

    def generate_anchors(self, width, height):
        """
        Generate anchors = centers of pixels.
        """
        # FIXME add batch_size?
        anchors = np.indices((width, height)).transpose((1, 2, 0))
        anchors = anchors + 0.5
        print("anchors shape=", tf.reshape(anchors, (-1, 2)).shape)
        #return anchors.reshape((-1, 2))
        return tf.reshape(tf.constant(anchors, dtype=tf.float32), (-1, 2))

    def clip_pixels(self, pixels):
        # Clip pixels to image boundaries
        # TODO
        #pixels[:, 0::2] = tf.maximum(tf.minimum(pixels[:, 0::2], tf.cast(im_shape[1] - 1, tf.float32)), 0.)
        #pixels[:, 1::2] = np.maximum(np.minimum(pixels[:, 1::2], im_shape[0] - 1), 0.)
        #pixels[:, 0::2] = np.maximum(np.minimum(pixels[:, 0::2], im_shape[1] - 1), 0.)
        #pixels[:, 1::2] = np.maximum(np.minimum(boxes[:, 1::2], im_shape[0] - 1), 0.)
        return pixels

    def pixels_transform_inv(self, pixels, deltas):
        #print(deltas.shape)
        # Given an anchor pixel and regression deltas, estimate proposal pixel
        #if pixels.shape[0] == 0:
        #    return tf.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

        #dx = deltas[:, 0::2]
        #dy = deltas[:, 1::2]

        #pred_pixels = tf.zeros_like(deltas)
        #pred_pixels[:, 0::2] = pixels[:, 0::2] + dx # FIXME shape
        #pred_pixels[:, 1::2] = pixels[:, 1::2] + dy # FIXME shape
        print("pixels shape=", pixels.shape)
        print("deltas shape=", deltas.shape)
        pred_pixels = pixels + deltas
        return pred_pixels

    def top_R_pixels(self, proposals, scores, R=20):
        """
        Order by score and take the top R proposals.
        Shapes are [N*N, 2] and [N*N, 1]
        """
        # Select top R pixel proposals if len(proposals) > R
        if proposals.get_shape().as_list()[0] > R:
            print(scores.shape)
            scores, keep = tf.nn.top_k(tf.squeeze(scores), k=R)
            print("Scores and keep:", scores.shape, keep.shape)
            print(proposals.shape)
            proposals = tf.gather(proposals, keep)
            print(proposals.shape)
        return proposals, scores

    def predicted_pixels(self, rpn_cls_prob, rpn_bbox_pred, anchors, R=20, classes=False):
        """
        rpn_cls_prob.shape = [None, N, N, n] where n = 2 (background/signal) or num_classes
        rpn_bbox_pred.shape = [None, N, N, 2]
        anchors.shape = [N*N, 2]
        Derive predicted pixels from predicted parameters (rpn_bbox_pred) with respect
        to the anchors (= centers of the pixels of the feature map).
        Return a list of predicted pixels and corresponding scores
        of shape [N*N, 2] and [N*N, n]
        """
        # Select pixels that contain something
        if classes:
            #scores = rpn_cls_prob[:, :, :, 2:]
            scores = tf.reshape(rpn_cls_prob, (-1, rpn_cls_prob.get_shape().as_list()[-1]))
        else:
            scores = rpn_cls_prob[:, :, :, 1:] # FIXME
            # Reshape to a list in the order of anchors
            # rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (-1, 2))
            scores = tf.reshape(scores, (-1, 1))

        # Get proposal pixels from regression deltas of rpn_bbox_pred
        #proposals = pixels_transform_inv(anchors, rpn_bbox_pred)
        print(anchors)
        anchors = tf.reshape(anchors, shape=(rpn_cls_prob.get_shape().as_list()[1], rpn_cls_prob.get_shape().as_list()[1], 2))
        print(rpn_bbox_pred, anchors)
        proposals =  anchors + rpn_bbox_pred
        proposals = tf.reshape(proposals, (-1, 2))
        # clip predicted pixels to the image
        proposals = self.clip_pixels(proposals)
        rois = tf.cast(proposals, tf.float32)
        return rois, scores

    def crop_pool_layer_2d(self, net, rois, R=20):
        """
        Crop and pool intermediate F3 layer.
        Net.shape = [1, 64, 64, 256]
        Rois.shape = [R, 2] # Could be less than R
        """
        print("\nnet shape=", net.shape)
        print("rois shape=", rois.shape)
        # Convert rois to normalized coordinates in [0, 1]
        rois = rois / 16.0
        # Shape of boxes = [num_boxes, 4]
        # boxes[i] is specified in normalized coordinates [y1, x1, y2, x2]
        # with y1 < y2 ideally
        boxes = tf.concat([rois, rois+4], axis=1)
        print("boxes shape=", boxes.shape)

        # Shape of box_ind = [num_boxes] with values in [0, batch_size)
        # FIXME allow batch size > 1
        box_ind = tf.fill((tf.shape(rois)[0],), 0)
        # 1-D tensor of 2 elements = [crop_height, crop_width]
        # All cropped image patches are resized to this size
        crop_size = tf.constant([64*2, 64*2])
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
