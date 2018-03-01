# *-* encoding: utf-8 *-*
# Unit tests for ppn functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
from ppn import PPN
from ppn_utils import generate_anchors, top_R_pixels, clip_pixels, compute_positives_ppn1, compute_positives_ppn2, assign_gt_pixels
from toydata_generator import ToydataGenerator

def generate_anchors_np(width, height, repeat=1):
    anchors_np = np.indices((width, height)).transpose((1, 2, 0))
    anchors_np = anchors_np + 0.5
    anchors_np = np.reshape(anchors_np, (-1, 2))
    return anchors_np

def predicted_pixels(rpn_cls_prob, rpn_bbox_pred, anchors, R=20, classes=False):
    """
    rpn_cls_prob.shape = [None, N, N, n] where n = 2 (background/signal) or num_classes
    rpn_bbox_pred.shape = [None, N, N, 2]
    anchors.shape = [N*N, 2]
    Derive predicted pixels from predicted parameters (rpn_bbox_pred) with respect
    to the anchors (= centers of the pixels of the feature map).
    Return a list of predicted pixels and corresponding scores
    of shape [N*N, 2] and [N*N, n]
    """
    with tf.variable_scope("predicted_pixels"):
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
        anchors = tf.reshape(anchors, shape=(-1, rpn_cls_prob.get_shape().as_list()[1], rpn_cls_prob.get_shape().as_list()[1], 2))
        proposals =  anchors + rpn_bbox_pred
        proposals = tf.reshape(proposals, (-1, 2))
        # clip predicted pixels to the image
        proposals = clip_pixels(proposals)
        rois = tf.cast(proposals, tf.float32)
        return rois, scores

class Test(unittest.TestCase):
    #self.toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
    #self.net = PPN()
    def test_generate_anchors(self):
        width, height = 2, 2
        repeat = 3
        anchors_np = generate_anchors_np(width=width, height=height, repeat=repeat)
        with tf.Session() as sess:
            anchors_tf = generate_anchors(width=width, height=height, repeat=repeat)
            return np.array_equal(anchors_tf, anchors_np)

    def test_top_R_pixels(self):
        R = 3
        threshold = 0.5
        # Shape N*N x 2
        proposals_np = np.array([[0.0, 1.0], [0.5, 0.7], [0.3, 0.88], [-0.2, 0.76], [0.23, 0.47], [0.33, 0.56], [0.0, 0.4], [-0.6, 0.3], [0.27, -0.98]])
        # Shape N*N x 1
        scores_np = np.array([0.1, 0.5, 0.7, 0.45, 0.65, 0.01, 0.78, 0.98, 0.72])
        threshold_indices = np.nonzero(scores_np > threshold)
        scores_np = scores_np[threshold_indices]
        proposals_np = proposals_np[threshold_indices]
        sorted_indices = np.argsort(scores_np)
        roi_scores_np = scores_np[sorted_indices][::-1][:R]
        rois_np = proposals_np[sorted_indices][::-1][:R]
        with tf.Session() as sess:
            proposals = tf.constant(proposals_np, dtype=tf.float32)
            scores =  tf.constant(scores_np, dtype=tf.float32)
            rois, roi_scores = top_R_pixels(proposals, scores, R=R, threshold=threshold)
            rois_tf, roi_scores_tf = sess.run([rois, roi_scores])
            return np.allclose(rois_tf, rois_np) and np.allclose(roi_scores_np, roi_scores_tf)

    def test_predicted_pixels1(self): # with classes=False ~ for PPN1
        R = 20
        width, height = 2, 2
        repeat = 1
        anchors_np = generate_anchors_np(width=width, height=height, repeat=repeat)
        # Shape [None, N, N, n] where n = 2 (background/signal) or num_classes
        rpn_cls_prob_np = np.array([[[[0.1, 0.9], [0.3, 0.7]], [[0.5, 0.5], [0.8, 0.2]]]])
        # Shape [None, N, N, 2]
        rpn_bbox_pred_np = np.array([[[[0.1, 0.1], [0.5, 0.2]], [[0.9, -0.5], [0.1, -0.4]]]])
        scores = rpn_cls_prob_np[:, :, :, 1:]
        roi_scores_np = np.reshape(scores, (-1, 1))
        anchors_np = np.reshape(anchors_np, (-1, rpn_cls_prob_np.shape[1], rpn_cls_prob_np.shape[1], 2))
        proposals =  anchors_np + rpn_bbox_pred_np
        proposals = np.reshape(proposals, (-1, 2))
        # clip predicted pixels to the image
        proposals = clip_pixels(proposals) # FIXME np function
        rois_np = proposals.astype(float)
        with tf.Session() as sess:
            anchors_tf = generate_anchors(width=width, height=height, repeat=repeat)
            rpn_cls_prob_tf = tf.constant(rpn_cls_prob_np, dtype=tf.float32)
            rpn_bbox_pred_tf = tf.constant(rpn_bbox_pred_np, dtype=tf.float32)
            rois, roi_scores = predicted_pixels(rpn_cls_prob_tf, rpn_bbox_pred_tf, anchors_tf, R=R, classes=False)
            rois_tf, roi_scores_tf = sess.run([rois, roi_scores])
            return np.allclose(rois_tf, rois_np) and np.allclose(roi_scores_tf, roi_scores_np)

    def test_predicted_pixels2(self): # with classes=True ~ for PPN2
        pass

    def test_include_gt_pixels(self):
        pass

    def test_compute_positives_ppn1(self):
        # Dummy input for testing, num of gt pixels = N = 3
        gt_pixels_test = np.array([[5.5, 7.7],[511.1, 433.3], [320, 320]])
        #print(gt_pixels_test.shape) #should be shape (3,2)

        classes_np = np.zeros((16,16))
        gt_pixels_np = np.floor(gt_pixels_test / 32.0).astype(int)
        gt_pixels_np = tuple(zip(*gt_pixels_np))
        classes_np[gt_pixels_np] = 1.
        classes_mask_np = classes_np.reshape(-1,1).astype(bool) # shape (16*16, 1)

        with tf.Session() as sess:
            classes_mask_tf = compute_positives_ppn1(gt_pixels_test)
            classes_mask_tf = sess.run([classes_mask_tf])
        return np.allclose(classes_mask_np, classes_mask_tf)

    def test_compute_positives_ppn2(self):
        # Need to comment out assert statement in compute_positives_ppn2
        # Dummy input for testing
        nb_rois, n, n_classes = 2, 3, 4
        scores_test = np.stack([np.array([0, 1, 0, 0]) for i in range(nb_rois*n*n)])
        closest_gt_distance_test = np.arange(nb_rois*n*n).reshape(nb_rois*n*n, 1)
        true_labels_test = np.ones((nb_rois*n*n, 1))
        thres_test = 20
        
        common_shape_np = np.array([nb_rois*n*n, 1])
        predicted_labels_np = np.argmax(scores_test, axis=1).reshape(common_shape_np)
        #print(predicted_labels_np.shape)
        mask_np = np.where(np.greater(closest_gt_distance_test, thres_test), False, True)
        mask_np = np.where(np.equal(true_labels_test, predicted_labels_np), mask_np, False)

        with tf.Session() as sess:
            mask_tf = compute_positives_ppn2(scores_test, closest_gt_distance_test, true_labels_test, threshold=thres_test)
            mask_tf = sess.run([mask_tf])
        return np.allclose(mask_np, mask_tf)
        
    def test_assign_gt_pixels(self):
        # Dummy input for testing
        N = 2 
        gt_pixels_placeholder_test = np.empty((N*N, 2))
        pass

if __name__ == '__main__':
    unittest.main()
