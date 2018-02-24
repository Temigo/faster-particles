# *-* encoding: utf-8 *-*
# Pixel Proposal Network
# Scratch implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

# Global parameters
R = 20
is_training = True
num_classes = 3 # background, track edge, shower start
width = 512
height = 512

def ppn():
    # Define placeholders
    image_placeholder       = tf.placeholder(name="image", shape=(1, None, None, 3), dtype=tf.float32)
    gt_pixels_placeholder   = tf.placeholder(name="gt_pixels", shape=(None, num_classes), dtype=tf.float32)
    input_shape_placeholder = tf.placeholder(name="input_shape", shape=(4,), dtype=tf.int32)

    # Define network
    with tf.variable_scope("vgg_16"):
        # VGG16 net
        net = slim.repeat(image_placeholder, 2, slim.conv2d, 64, [3, 3],
                          trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                        trainable=False, scope='conv2')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                        trainable=is_training, scope='conv3')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv4')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                        trainable=is_training, scope='conv5')

        # --- Pixel Proposal Network 1 ---
    with tf.variable_scope("ppn1"):
        # Define initializers
        rcnn_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # Step 0) Convolution for RPN/Detection shared layer
        rpn = slim.conv2d(net,
                          512, # RPN Channels = num_outputs
                          (3, 3), # RPN Kernels : (3, 3)
                          trainable=True,
                          weights_initializer=rcnn_initializer,
                          scope="rpn_conv/3x3")
        # Step 1-a) PPN 2 pixel position predictions
        rpn_bbox_pred = slim.conv2d(rpn, 2, [1, 1],
                                    trainable=True,
                                    weights_initializer=rcnn_initializer,
                                    padding='VALID',
                                    activation_fn=None,
                                    scope='rpn_bbox_pred')
        # Step 1-b) Generate 2 class scores
        rpn_cls_score = slim.conv2d(rpn, 2, [1, 1],
                                    trainable=True,
                                    weights_initializer=rcnn_initializer,
                                    padding='VALID',
                                    activation_fn=None,
                                    scope='rpn_cls_score')
        # Compute softmax
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score) # FIXME might need a reshape here

        # Step 3) Get a (meaningful) subset of rois and associated scores
        anchors = generate_anchors(width, height)
        print(anchors)
        # Step 3-a) Derive predicted bbox (rois) with scores (roi_scores) from prediction parameters (rpn_bbox_pred)
        #           and anchors. Some boxes are filtered out based on NMS of proposed regions and objectness
        #           probability (rpn_cls_prob)
        rois, roi_scores = proposal_layer_2d(rpn_cls_prob, rpn_bbox_pred, anchors, input_shape_placeholder)

"""        # Step 3-b) Map RPN labels to ground-truth boxes. rpn_labels.size == total # of anchors
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer_2d()

        # Step 3-c) Anchor rois and roi_scores with ground truth
        with tf.control_dependencies([rpn_labels]):
            rois, _ = proposal_target_layer_2d(rois, roi_scores)

        # --- END of Pixel Proposal Network 1 ---

        # Pool to Pixels of Interest
        rpn_pooling = crop_pool_layer_2d(net, rois, "rpn_pooling")

        # --- Pixel Proposal Network 2 ---
    with tf.variable_scope("ppn2"):
        # Define initializers
        rcnn_initializer2=tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        initializer2 = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        initializer_bbox2 = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # Step 0) Convolution for RPN/Detection shared layer
        rpn2 = slim.conv2d(rpn_pooling,
                          512, # RPN Channels = num_outputs
                          (3, 3), # RPN Kernels : (3, 3) -> (1, 1) ?
                          trainable=True,
                          weights_initializer=rcnn_initializer2,
                          scope="rpn_conv2/3x3")
        # Step 1-a) PPN 2 pixel prediction parameters
        rpn_bbox_pred2 = slim.conv2d(rpn2, 2, [1, 1],
                                    trainable=True,
                                    weights_initializer=rcnn_initializer2,
                                    padding='VALID',
                                    activation_fn=None,
                                    scope='rpn_bbox_pred2')
        # Step 1-b) Generate 2 class scores
        rpn_cls_score2 = slim.conv2d(rpn2, 2, [1, 1],
                                    trainable=True,
                                    weights_initializer=rcnn_initializer2,
                                    padding='VALID',
                                    activation_fn=None,
                                    scope='rpn_cls_score2')
        # Compute softmax
        rpn_cls_prob2 = tf.nn.softmax(rpn_cls_score2) # FIXME might need a reshape here

        # Step 3) Get a (meaningful) subset of rois and associated scores
        # Step 3-a) Derive predicted bbox (rois) with scores (roi_scores) from prediction parameters (rpn_bbox_pred)
        #           and anchors. Some boxes are filtered out based on NMS of proposed regions and objectness
        #           probability (rpn_cls_prob)
        rois2, roi_scores2 = proposal_layer_2d(rpn_cls_prob2, rpn_bbox_pred2)

        # Step 3-b) Map RPN labels to ground-truth boxes. rpn_labels.size == total # of anchors
        rpn_labels2, rpn_bbox_targets2, rpn_bbox_inside_weights2, rpn_bbox_outside_weights2 = anchor_target_layer_2d()

        # Step 3-c) Anchor rois and roi_scores with ground truth
        with tf.control_dependencies([rpn_labels2]):
            rois2, _ = proposal_target_layer_2d(rois2, roi_scores2)


        # --- END of Pixel Proposal Network 2 ---

        # Pool to Pixels of Interest
        rpn_pooling2 = crop_pool_layer_2d(net, rois2, "rpn_pooling") # FIXME net?

        # --- Pixel classification ---
        net_flat = slim.flatten(rpn_pooling2, scope='flatten')
        fc6 = slim.fully_connected(net_flat, 4096, scope='fc6')
        if is_training:
            fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                scope='dropout6')
        rcnn_input = slim.fully_connected(fc6, 4096, scope='fc7')
        if is_training:
            rcnn_input = slim.dropout(rcnn_input, keep_prob=0.5, is_training=True,
                                scope='dropout7')
        #cls_prob, bbox_pred = region_classification_2d(rcnn_input, trainable,
        #                                                     initializer, initializer_bbox)


"""
def generate_anchors(width, height):
    anchors = np.indices((width, height)).transpose((1, 2, 0))
    return anchors.reshape((-1, 2))

def clip_pixels(pixels, im_shape):
    # Clip pixels to image boundaries
    # TODO
    #pixels[:, 0::2] = tf.maximum(tf.minimum(pixels[:, 0::2], tf.cast(im_shape[1] - 1, tf.float32)), 0.)
    #pixels[:, 1::2] = np.maximum(np.minimum(pixels[:, 1::2], im_shape[0] - 1), 0.)
    #pixels[:, 0::2] = np.maximum(np.minimum(pixels[:, 0::2], im_shape[1] - 1), 0.)
    #pixels[:, 1::2] = np.maximum(np.minimum(boxes[:, 1::2], im_shape[0] - 1), 0.)
    return pixels

def pixels_transform_inv(pixels, deltas):
    #print(deltas.shape)
    # Given an anchor pixel and regression deltas, estimate proposal pixel
    #if pixels.shape[0] == 0:
    #    return tf.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    #dx = deltas[:, 0::2]
    #dy = deltas[:, 1::2]

    #pred_pixels = tf.zeros_like(deltas)
    #pred_pixels[:, 0::2] = pixels[:, 0::2] + dx # FIXME shape
    #pred_pixels[:, 1::2] = pixels[:, 1::2] + dy # FIXME shape
    pred_pixels = pixels + deltas
    return pred_pixels

def proposal_layer_2d(rpn_cls_prob, rpn_bbox_pred, anchors, input_shape, R=20):
    # Select pixels that contain something
    scores = rpn_cls_prob[:, :, :, 1:] # FIXME
    # Reshape to a list in the order of anchors
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, (-1, 2))
    scores = tf.reshape(scores, (-1, 1))

    # Get proposal pixels from regression deltas of rpn_bbox_pred
    proposals = pixels_transform_inv(anchors, rpn_bbox_pred)
    # clip predicted pixels to the image
    proposals = clip_pixels(proposals, input_shape)
    # Select top R pixel proposals
    print(scores.shape)
    scores, keep = tf.nn.top_k(tf.squeeze(scores), k=R)
    print(scores.shape, keep.shape)
    print(proposals.shape)
    proposals = proposals[keep]
    print(proposals.shape)
    # Expand the array via np.hstack and zero-filled array to later store an object class prediction.
    # here we assume only-1-image-batch-size to initialize the array size
    batch_inds = tf.zeros((proposals.shape[0], 1), dtype=tf.float32)
    rois = tf.concat([batch_inds, tf.cast(proposals, tf.float32)], axis=1)

    tf.reshape(rois, ([None, 5]))
    tf.reshape(scores, ([None, 1]))

    return rois, scores

"""
def anchor_target_layer_2d(gt_pixels, all_anchors):
    # Assign foreground/background label to each anchor

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(all_anchors),), dtype=np.float32)
    labels.fill(-1)

    rpn_labels.set_shape([1, 1, None, None])
    rpn_bbox_targets.set_shape([1, None, None, 2])
    rpn_bbox_inside_weights.set_shape([1, None, None, 2]) # What are they for?
    rpn_bbox_outside_weights.set_shape([1, None, None, 2])

    rpn_labels = tf.to_int32(rpn_labels, name="to_int32")

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
"""
if __name__ == "__main__":
    #net = ppn()
    ppn()
    # Dummy 4x4 image
    #dummy_rpn_cls_prob = np.ndarray([[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]]])
    #dummy_rpn_bbox_pred = np.ndarray([])
    #dummy_anchors =
    #dummy_input_shape =
    #proposal_layer_2d(dummy_rpn_cls_prob, dummy_rpn_bbox_pred, dummy_anchors, dummy_input_shape)

    image = tf.placeholder(tf.float32,[1,512,512,3])
    net.set_input_shape(image)
    # Create a session
    sess = tf.InteractiveSession()
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    #ret = sess.run(net._anchors,feed_dict={})
    #print('{:s}'.format(ret))
