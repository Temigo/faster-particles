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
        self.ppn1_score_threshold = 0.5
        self.ppn2_distance_threshold = 2

    def create_architecture(self):
        # Define placeholders
        # FIXME Assuming batch size of 1 currently
        self.image_placeholder       = tf.placeholder(name="image", shape=(1, 512, 512, 3), dtype=tf.float32)
        # Shape of gt_pixels_placeholder = nb_gt_pixels, 2 coordinates + 1 class label in [0, num_classes)
        self.gt_pixels_placeholder   = tf.placeholder(name="gt_pixels", shape=(None, 3), dtype=tf.float32)

        # Define network regularizers
        weights_regularizer = tf.contrib.layers.l2_regularizer(0.0005)
        biases_regularizer = tf.no_regularizer
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=weights_regularizer,
                            biases_regularizer=biases_regularizer,
                            biases_initializer=tf.constant_initializer(0.0)):
            # =====================================================
            # --- VGG16 net = 13 conv layers with 5 max-pooling ---
            # =====================================================
            with tf.variable_scope("vgg_16"):
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

            # =====================================================
            # ---       Pixel Proposal Network 1                ---
            # =====================================================
            with tf.variable_scope("ppn1"):
                # Define initializers
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

                # Step 0) Convolution for RPN/Detection shared layer
                # Shape of rpn = 1, 16, 16, 512
                ppn1 = slim.conv2d(net2,
                                  512, # RPN Channels = num_outputs
                                  (3, 3), # RPN Kernels : (3, 3)
                                  trainable=True,
                                  weights_initializer=initializer,
                                  scope="ppn1_conv/3x3")
                # Step 1-a) PPN 2 pixel position predictions
                # Shape of rpn_bbox_pred = 1, 16, 16, 2
                ppn1_pixel_pred = slim.conv2d(ppn1, 2, [1, 1],
                                            trainable=True,
                                            weights_initializer=initializer,
                                            padding='VALID',
                                            activation_fn=None,
                                            scope='ppn1_pixel_pred')
                # Step 1-b) Generate 2 class scores (background vs signal)
                # Shape of rpn_cls_score = 1, 16, 16, 2
                # FIXME use sigmoid instead of softmax?
                ppn1_cls_score = slim.conv2d(ppn1, 2, [1, 1],
                                            trainable=True,
                                            weights_initializer=initializer,
                                            padding='VALID',
                                            activation_fn=None,
                                            scope='ppn1_cls_score')

                # Compute softmax
                # Shape of rpn_cls_prob = 1, 16, 16, 2
                ppn1_cls_prob = tf.nn.softmax(ppn1_cls_score)
                # print("rpn_cls_prob shape:", rpn_cls_prob.shape)

                # Step 3) Get a (meaningful) subset of rois and associated scores
                # Generate anchors = pixel centers of the last feature map.
                # Shape of anchors = 16*16, 2
                anchors = self.generate_anchors(width=16, height=16) # FIXME express width and height better
                assert anchors.get_shape().as_list() == [256, 2]

                # Derive predicted positions (poi) with scores (poi_scores) from prediction parameters
                # and anchors. Take the first R proposed pixels which contain an object.
                proposals, scores = self.predicted_pixels(ppn1_cls_prob, ppn1_pixel_pred, anchors)
                rois, roi_scores = self.top_R_pixels(proposals, scores, R=20, threshold=self.ppn1_score_threshold)
                assert proposals.get_shape().as_list() == [256, 2]
                assert scores.get_shape().as_list() == [256, 1]
                assert rois.get_shape().as_list() == [None, 2]
                assert roi_scores.get_shape().as_list() == [None, 1]

                # all outputs from 1x1 convolution are categorized into “positives” and “negatives”.
                # Positives = pixels which contain a ground-truth point
                # Negatives = other pixels
                classes_mask = self.compute_positives_ppn1()
                assert classes_mask.get_shape().as_list() == [256, 1]
                # FIXME Use Kazu's pixel index to limit the number of gt points for
                # which we compute a distance from a unique proposed point per pixel.

                # For each pixel of the F5 features map get distance between proposed point
                # and the closest ground truth pixel
                # Don't forget to convert gt pixels coordinates to F5 coordinates
                closest_gt, closest_gt_distance, _ = self.assign_gt_pixels(proposals)
                assert closest_gt.get_shape().as_list() == [256]
                assert closest_gt_distance.get_shape().as_list() == [256, 1]
                #assert closest_gt_label.get_shape().as_list() == [256, 1]

                # Step 4) compute loss for PPN1
                # First is point loss: for positive pixels, distance from proposed pixel to closest ground truth pixel
                # FIXME reduce_mean or reduce_sum? Same for loss_ppn1_class
                loss_ppn1_point = tf.reduce_mean(tf.boolean_mask(closest_gt_distance, classes_mask))
                # Use softmax_cross_entropy instead of sigmoid here
                #loss_ppn1_class = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(classes_mask, tf.float32), logits=scores))
                loss_ppn1_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(classes_mask, (-1,)), tf.int32),
                                                                                                logits=tf.reshape(ppn1_cls_score, (-1, 2))))
            # --- END of Pixel Proposal Network 1 ---

            # Check if all ground truth pixels are covered by ROIs
            # If not, add relevant ROIs on F3
            # TODO Algorithm should not place 4x4 exactly centered around ground-truth point,
            # but instead allow random variation
            rois = self.include_gt_pixels(rois)
            assert rois.get_shape().as_list() == [None, 2]

            # Pool to Pixels of Interest of intermediate layer
            # FIXME How do we want to do the ROI pooling?
            # Shape of rpn_pooling = nb_rois, 4, 4, 256
            rpn_pooling = self.crop_pool_layer_2d(net, rois)
            assert rpn_pooling.get_shape().as_list() == [None, 4, 4, 256]

            # =====================================================
            # ---         Pixel Proposal Network 2              ---
            # =====================================================
            with tf.variable_scope("ppn2"):
                # Define initializers
                initializer2=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

                batch_size = tf.shape(rpn_pooling)[0]
                # Step 0) Convolution for PPN2 intermediate layer
                # Based on F3 feature map (ie after 3 max-pool layers in VGG)
                # Shape = nb_rois, 4, 4, 512
                ppn2 = slim.conv2d(rpn_pooling,
                                  512, # RPN Channels = num_outputs
                                  (3, 3), # RPN Kernels : (3, 3)
                                  trainable=True,
                                  weights_initializer=initializer2,
                                  scope="ppn2_conv/3x3")
                # Step 1-a) PPN 2 pixel prediction parameters
                # Proposes pixel position (x, y) w.r.t. pixel center = anchor
                # Shape of rpn_bbox_pred2 = nb_rois, 4, 4, 2
                ppn2_pixel_pred = slim.conv2d(ppn2, 2, [1, 1],
                                            trainable=True,
                                            weights_initializer=initializer2,
                                            padding='VALID',
                                            activation_fn=None,
                                            scope='ppn2_pixel_pred2')
                # print("rpn_bbox_pred2 shape=", rpn_bbox_pred2.shape)
                # Step 1-b) Generate class scores
                # Shape of rpn_cls_score2 = nb_rois, 4, 4, num_classes
                ppn2_cls_score = slim.conv2d(ppn2, self.num_classes, [1, 1],
                                            trainable=True,
                                            weights_initializer=initializer2,
                                            padding='VALID',
                                            activation_fn=None,
                                            scope='ppn2_cls_score')
                # print("rpn_cls_score2 shape=", rpn_cls_score2.shape)
                # Compute softmax
                ppn2_cls_prob = tf.nn.softmax(ppn2_cls_score) # FIXME might need a reshape here
                # print("rpn_cls_prob2 shape=", rpn_cls_prob2.shape)

                # Step 3) Get a (meaningful) subset of rois and associated scores
                # Anchors are defined as center of pixels
                # Shape [nb_rois * 4 * 4 , 2]
                anchors2 = self.generate_anchors(width=4, height=4, repeat=batch_size) # FIXME express width and height better
                assert anchors2.get_shape().as_list() == [None, 2]
                # Derive proposed points from delta predictions (rpn_bbox_pred2) w.r.t. pixels centers
                # Coordinates of proposals2 are in 4x4 ROI area
                # We have 4*4*num_roi proposals and corresponding scores
                proposals2, scores2 = self.predicted_pixels(ppn2_cls_prob, ppn2_pixel_pred, anchors2, classes=True)
                assert proposals2.get_shape().as_list() == [None, 2]
                assert scores2.get_shape().as_list() == [None, self.num_classes]
                # Find closest ground truth pixel and its label
                # Option roi allows to convert gt_pixels_placeholder information to ROI 4x4 coordinates
                closest_gt, closest_gt_distance, true_labels = self.assign_gt_pixels(proposals2, rois=rois)
                assert closest_gt.get_shape().as_list() == [None]
                assert closest_gt_distance.get_shape().as_list() == [None, 1]
                assert true_labels.get_shape().as_list() == [None, 1]

                # Positives now = pixels within certain distance range from
                # the closest ground-truth point of the same class (track edge or shower start)
                positives = self.compute_positives_ppn2(scores2, closest_gt_distance, true_labels, threshold=self.ppn2_distance_threshold)
                assert positives.get_shape().as_list() == [None, 1]

                # Step 4) Loss
                # first is based on an absolute distance to the closest
                # ground-truth point where only positives count
                loss_ppn2_point = tf.reduce_sum(tf.boolean_mask(closest_gt_distance, positives))
                # second is a softmax class loss from both positives and negatives
                # for positives, the true label is defined by the closest point’s label
                loss_ppn2_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(tf.reshape(true_labels, (-1,)), tf.int32),
                                                                                 logits=tf.reshape(ppn2_cls_score, (-1, self.num_classes)))

                # --- END of Pixel Proposal Network 2 ---

                # Turn predicted positions (float) into original image positions
                # Convert proposals2 ROI 4x4 coordinates to 64x64 F3 coordinates
                # then back to original image.
                # f3_proposals = tf.reshape(tf.boolean_mask(proposals2, positives), (-1, 16, 2))
                # im_proposals = (f3_proposals + 4*rois)*8.0
                # im_labels = tf.argmax(tf.boolean_mask(scores2, positives), axis=1)
                # We have now 4*4*num_roi proposals and corresponding labels in original image.
                # Pixel NMS equivalent ?

    def get_gt_pixels(self):
        """
        Slice first 2 dimensions of gt_pixels_placeholder (coordinates only)
        We want it to be shape (None, 2)
        """
        # FIXME check that this returns the expected
        # return tf.squeeze(self.gt_pixels_placeholder, axis=[2])
        return tf.slice(self.gt_pixels_placeholder, [0, 0], [-1, 2])

    def include_gt_pixels(self, rois):
        """
        Rois: [None, 2] in F5 coordinates
        Return rois in F5 coordinates
        """
        # Slice first 2 dimensions of gt_pixels_placeholder
        # We want it to be shape (None, 2)
        gt_pixels = self.get_gt_pixels()
        # convert to F5 coordinates
        gt_pixels_coord = tf.cast(tf.floor(gt_pixels / 32.0), tf.float32)
        # FIXME As soon as new version of Tensorflow supporting axis option
        # for tf.unique, replace the following rough patch.
        # In the meantime, we will have some duplicates between rois and gt_pixels.
        rois = tf.concat([rois, gt_pixels_coord], axis=0) # shape [None, 2]
        #rois = tf.sets.set_union(tf.cast(rois, tf.int32), tf.cast(gt_pixels_coord, tf.int32))
        #rois, _, _ = tf.unique_with_counts(rois)
        #print(rois)
        assert rois.get_shape().as_list() == [None, 2]
        return rois

    def compute_positives_ppn2(self, scores, closest_gt_distance, true_labels, threshold=2):
        """
        closest_gt_distance shape = (A*N*N, 1)
        true_labels shape = (A*N*N, 1)
        scores shape = (A*N*N, num_classes)
        Return boolean mask for positives among proposals.
        Positives are those within certain distance range from the
        closest ground-truth point of the same class
        """
        pixel_count = tf.shape(true_labels)[0]
        common_shape = tf.stack([pixel_count, 1])
        predicted_labels = tf.reshape(tf.argmax(scores, axis=1, output_type=tf.int32), common_shape)
        assert predicted_labels.get_shape().as_list() == [None, 1]
        true_labels = tf.cast(true_labels, tf.int32)
        mask = tf.where(tf.greater(closest_gt_distance, threshold), tf.fill(common_shape, False), tf.fill(common_shape, True))
        mask = tf.where(tf.equal(true_labels, predicted_labels), mask, tf.fill(common_shape, False))
        return mask

    def assign_gt_pixels(self, proposals, rois=None):
        """
        Proposals shape: [A*N*N, 2] (N=16 or 64)
        gt_pixels_full is shape [None, 2, 1]
        Classes shape: [A*N*N, 1]
        Rois shape: [A, 2] coordinates in F5 feature map (16x16)
        Option roi allows to convert gt_pixels_placeholder information to ROI 4x4 coordinates
        Returns closest ground truth pixels for all pixels and corresponding distance
        """
        # Slice first 2 dimensions of gt_pixels_placeholder
        # We want it to be shape (None, 2)
        gt_pixels = self.get_gt_pixels()
        gt_pixels = tf.expand_dims(gt_pixels, axis=0)
        if rois is None:
            # Tile to have shape (A*N*N, None, 2)
            gt_pixels = gt_pixels / 32.0 # Convert to F5 coordinates
            all_gt_pixels = tf.tile(gt_pixels, tf.stack([tf.shape(proposals)[0], 1, 1]))

        else: # Translate each batch of N*N rows of all_gt_pixels w.r.t. corresponding ROI center
            # FIXME check that this yields expected result
            # Translation is gt_pixels / 8.0 - 4*rois[i] (with conversion to F3 coordinates)
            # Go to shape [1, 1, None, 2]
            gt_pixels = tf.expand_dims(gt_pixels, axis=0)
            # Tile to shape [A, N*N, None, 2]
            gt_pixels = tf.tile(gt_pixels, [tf.shape(rois)[0], tf.cast(tf.shape(proposals)[0]/tf.shape(rois)[0], tf.int32), 1, 1])
            # Broadcast translation
            all_gt_pixels = gt_pixels / 8.0 - 4.0 * rois
            # Reshape to [A*N*N, None, 2]
            all_gt_pixels = tf.reshape(all_gt_pixels, (tf.shape(proposals)[0], -1, 2))

        assert all_gt_pixels.get_shape().as_list() == [None, None, 2]
        # Reshape proposals to [A*N*N, 1, 2]
        proposals = tf.expand_dims(proposals, axis=1)
        distances = tf.sqrt(tf.reduce_sum(tf.pow(proposals - all_gt_pixels, 2), axis=2))
        # distances.shape = [A*N*N, None]
        # closest_gt.shape = [A*N*N,]
        # closest_gt[i] = indice of closest gt in gt_pixels_placeholder
        closest_gt = tf.argmin(distances, axis=1)
        closest_gt_distance = tf.reduce_min(distances, axis=1, keep_dims=True)
        #print("squeezed gt_pixels_placeholder shape=", tf.squeeze(tf.slice(gt_pixels_placeholder, [0,0,0], [-1,1,-1]), axis=1).shape)
        closest_gt_label = tf.nn.embedding_lookup(tf.slice(self.gt_pixels_placeholder, [0, 2], [-1, 1]), closest_gt)
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
        gt_pixels = self.get_gt_pixels()
        # Convert to F5 coordinates (16x16)
        # Shape = None, 2
        gt_pixels = tf.cast(tf.floor(gt_pixels / 32.0), tf.int32)
        # Assign positive pixels based on gt_pixels
        #classes = classes + tf.scatter_nd(gt_pixels, tf.constant(value=1.0, shape=tf.shape(gt_pixels)[0]), classes.shape)
        classes = classes + tf.scatter_nd(gt_pixels, tf.fill((tf.shape(gt_pixels)[0],), 1.0), classes.shape)
        classes = tf.cast(tf.reshape(classes, shape=(-1, 1)), tf.int32)
        classes_mask = tf.cast(classes, tf.bool) # Turn classes into a mask
        return classes_mask

    def generate_anchors(self, width, height, repeat=1):
        """
        Generate anchors = centers of pixels.
        Repeat ~ batch size.
        """
        anchors = np.indices((width, height)).transpose((1, 2, 0))
        anchors = anchors + 0.5
        anchors = tf.reshape(tf.constant(anchors, dtype=tf.float32), (-1, 2))
        return tf.tile(anchors, tf.stack([repeat, 1]))

    def clip_pixels(self, pixels):
        # TODO Clip pixels to image boundaries
        #pixels[:, 0::2] = tf.maximum(tf.minimum(pixels[:, 0::2], tf.cast(im_shape[1] - 1, tf.float32)), 0.)
        #pixels[:, 1::2] = np.maximum(np.minimum(pixels[:, 1::2], im_shape[0] - 1), 0.)
        #pixels[:, 0::2] = np.maximum(np.minimum(pixels[:, 0::2], im_shape[1] - 1), 0.)
        #pixels[:, 1::2] = np.maximum(np.minimum(boxes[:, 1::2], im_shape[0] - 1), 0.)
        return pixels

    def pixels_transform_inv(self, pixels, deltas):
        # Given an anchor pixel and regression deltas, estimate proposal pixel
        print("pixels shape=", pixels.shape)
        print("deltas shape=", deltas.shape)
        pred_pixels = pixels + deltas
        return pred_pixels

    def top_R_pixels(self, proposals, scores, R=20, threshold=0.5):
        """
        Order by score and take the top R proposals.
        Shapes are [N*N, 2] and [N*N, 1]
        """
        # Select top R pixel proposals
        flat_scores = tf.squeeze(scores) # shape N*N
        R = min(R, flat_scores.get_shape().as_list()[0])
        # Output of tf.nn.top_k will be sorted in descending order
        scores, keep = tf.nn.top_k(tf.squeeze(scores), k=R, sorted=True)
        assert scores.get_shape().as_list() == [R]
        assert keep.get_shape().as_list() == [R]
        # Select scores above threshold
        keep2 = tf.where(tf.greater(scores, threshold))
        assert keep2.get_shape().as_list() == [None, 1]
        proposals = tf.gather(tf.gather(proposals, keep), tf.reshape(keep2, (-1,)))
        scores = tf.gather(scores, keep2)
        assert proposals.get_shape().as_list() == [None, 2]
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
        anchors = tf.reshape(anchors, shape=(rpn_cls_prob.get_shape().as_list()[1], rpn_cls_prob.get_shape().as_list()[1], 2))
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
        Rois.shape = [None, 2] # Could be less than R
        """
        assert net.get_shape().as_list() == [1, 64, 64, 256]
        assert rois.get_shape().as_list() == [None, 2]
        # Convert rois to normalized coordinates in [0, 1] of 64x64 feature map
        rois = rois / 16.0
        # Shape of boxes = [num_boxes, 4]
        # boxes[i] is specified in normalized coordinates [y1, x1, y2, x2]
        # with y1 < y2 ideally
        boxes = tf.concat([rois, rois+4], axis=1)
        assert boxes.get_shape().as_list() == [None, 4]

        # Shape of box_ind = [num_boxes] with values in [0, batch_size)
        # FIXME allow batch size > 1
        box_ind = tf.fill((tf.shape(rois)[0],), 0)
        # 1-D tensor of 2 elements = [crop_height, crop_width]
        # All cropped image patches are resized to this size
        crop_size = tf.constant([4*2, 4*2])
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
