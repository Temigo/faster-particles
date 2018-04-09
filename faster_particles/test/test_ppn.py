# *-* encoding: utf-8 *-*
# Unit tests for ppn functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
from faster_particles.ppn import PPN
from faster_particles.ppn_utils import generate_anchors, top_R_pixels, clip_pixels, \
    compute_positives_ppn1, compute_positives_ppn2, assign_gt_pixels, \
    include_gt_pixels, predicted_pixels
from faster_particles.toydata.toydata_generator import ToydataGenerator

def generate_anchors_np(width, height, repeat=1):
    anchors_np = np.indices((width, height)).transpose((1, 2, 0))
    anchors_np = anchors_np + 0.5
    anchors_np = np.reshape(anchors_np, (-1, 2))
    return anchors_np

def clip_pixels_np(pixels, im_shape):
    """
    pixels shape: [None, 2]
    Clip pixels (x, y) to [0, im_shape[0]) x [0, im_shape[1])
    """
    pixels[:, 0] = np.clip(pixels[:, 0], 0, im_shape[0])
    pixels[:, 1] = np.clip(pixels[:, 1], 0, im_shape[1])
    return pixels

class Test(unittest.TestCase):
    #self.toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
    #self.net = PPN()
    def test_generate_anchors(self):
        width, height = 2, 2
        repeat = 3
        anchors_np = generate_anchors_np(width=width, height=height, repeat=repeat)
        with tf.Session() as sess:
            anchors_tf = generate_anchors((width, height), repeat=repeat)
            return np.array_equal(anchors_tf, anchors_np)

    def test_clip_pixels(self):
        im_shape = (3, 3)
        proposals_np = np.array([[-0.5, 1.0], [0.01, 3.4], [2.5, 2.99]])
        pixels_np = clip_pixels_np(proposals_np, im_shape)
        with tf.Session() as sess:
            proposals = tf.constant(proposals_np, dtype=tf.float32)
            pixels = clip_pixels(proposals, im_shape)
            pixels_tf = sess.run(pixels)
            return np.allclose(pixels_np, pixels_tf)

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

        # Shape [None, N, N, n] where n = 2 (background/signal)
        rpn_cls_prob_np = np.array([[[[0.1, 0.9], [0.3, 0.7]], [[0.5, 0.5], [0.8, 0.2]]]])
        # Shape [None, N, N, 2]
        rpn_bbox_pred_np = np.array([[[[0.1, 0.1], [0.5, 0.2]], [[0.9, -0.5], [0.1, -0.4]]]])

        anchors_np = generate_anchors_np(width=width, height=height, repeat=repeat)
        scores = rpn_cls_prob_np[:, :, :, 1:]
        roi_scores_np = np.reshape(scores, (-1, 1))
        anchors_np = np.reshape(anchors_np, (-1, rpn_cls_prob_np.shape[1], rpn_cls_prob_np.shape[1], 2))
        proposals =  anchors_np + rpn_bbox_pred_np
        proposals = np.reshape(proposals, (-1, 2))
        # clip predicted pixels to the image
        proposals = clip_pixels_np(proposals, (width, height)) # FIXME np function
        rois_np = proposals.astype(float)

        with tf.Session() as sess:
            anchors_tf = generate_anchors(width=width, height=height, repeat=repeat)
            rpn_cls_prob_tf = tf.constant(rpn_cls_prob_np, dtype=tf.float32)
            rpn_bbox_pred_tf = tf.constant(rpn_bbox_pred_np, dtype=tf.float32)
            rois, roi_scores = predicted_pixels(rpn_cls_prob_tf, rpn_bbox_pred_tf, anchors_tf, (width, height), R=R, classes=False)
            rois_tf, roi_scores_tf = sess.run([rois, roi_scores])
            return np.allclose(rois_tf, rois_np) and np.allclose(roi_scores_tf, roi_scores_np)

    def test_predicted_pixels2(self): # with classes=True ~ for PPN2
        R = 20
        width, height = 2, 2
        repeat = 1

        # Shape [None, N, N, n] where n = num_classes
        rpn_cls_prob_np = np.array([[[[0.1, 0.8, 0.1], [0.3, 0.65, 0.05]], [[0.5, 0.02, 0.48], [0.8, 0.18, 0.02]]]])
        # Shape [None, N, N, 2]
        rpn_bbox_pred_np = np.array([[[[0.1, 0.1], [0.5, 0.2]], [[0.9, -0.5], [0.1, -0.4]]]])

        rpn_cls_prob_np = rpn_cls_prob_np[:,:,:,1:]
        anchors_np = generate_anchors_np(width=width, height=height, repeat=repeat)
        roi_scores_np = np.reshape(rpn_cls_prob_np, (-1, rpn_cls_prob_np.shape[-1]))

        anchors_np = np.reshape(anchors_np, (-1, rpn_cls_prob_np.shape[1], rpn_cls_prob_np.shape[1], 2))
        proposals =  anchors_np + rpn_bbox_pred_np
        proposals = np.reshape(proposals, (-1, 2))
        # clip predicted pixels to the image
        proposals = clip_pixels_np(proposals, (width, height)) # FIXME np function
        rois_np = proposals.astype(float)

        with tf.Session() as sess:
            anchors_tf = generate_anchors(width=width, height=height, repeat=repeat)
            rpn_cls_prob_tf = tf.constant(rpn_cls_prob_np, dtype=tf.float32)
            rpn_bbox_pred_tf = tf.constant(rpn_bbox_pred_np, dtype=tf.float32)
            rois, roi_scores = predicted_pixels(rpn_cls_prob_tf, rpn_bbox_pred_tf, anchors_tf, (width, height), R=R, classes=True)
            rois_tf, roi_scores_tf = sess.run([rois, roi_scores])
            return np.allclose(rois_tf, rois_np) and np.allclose(roi_scores_tf, roi_scores_np)

    def test_include_gt_pixels(self):
        # [None, 2] in F5 coordinates
        rois_np = np.array([[0, 3], [15, 2], [3, 4], [5.6, 9.1]])
        # [None, 2]
        gt_pixels_np = np.array([[2.4, 2.3], [3, 4], [6.4, 1.2]])

        # convert to F3 coordinates
        gt_pixels_coord = np.floor(gt_pixels_np / 8.0)
        # Get 3x3 pixels around this in F3
        gt_pixels_coord = gt_pixels_coord[:, np.newaxis, :]
        gt_pixels_coord = np.tile(gt_pixels_coord, [1, 9, 1]) # shape N x 9 x 2
        update = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
        update = np.tile(update[np.newaxis, :, :], [gt_pixels_coord.shape[0], 1, 1])
        gt_pixels_coord = gt_pixels_coord + update
        gt_pixels_coord = np.reshape(gt_pixels_coord, (-1, 2)) # Shape N*9, 2
        # FIXME Clip it to F3 size
        # indices = tf.where(tf.less(gt_pixels_coord, 64))
        # gt_pixels_coord = tf.gather_nd(gt_pixels_coord, indices)
        # Go back to F5 coordinates
        gt_pixels_coord = gt_pixels_coord / 4.0 # FIXME hardcoded
        rois_result_np = np.vstack([np.floor(rois_np), gt_pixels_coord]) # shape [None, 2]

        with tf.Session() as sess:
            rois_tf = tf.constant(rois_np, dtype=tf.float32)
            gt_pixels_tf = tf.constant(gt_pixels_np, dtype=tf.float32)
            rois_tf = include_gt_pixels(rois_tf, gt_pixels_tf)
            rois_result_tf = sess.run(rois_tf)
            return np.allclose(rois_result_tf, rois_result_np)

    def test_compute_positives_ppn1(self):
        # Dummy input for testing, num of gt pixels = N = 3
        gt_pixels_test = np.array([[5.5, 7.7], [511.1, 433.3], [320, 320]])
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
        nb_rois, N, n_classes = 2, 3, 4
        scores_test = np.stack([np.array([0, 1, 0, 0]) for i in range(nb_rois*N*N)])
        closest_gt_distance_test = np.arange(nb_rois*N*N).reshape(nb_rois*N*N, 1)
        true_labels_test = np.ones((nb_rois*N*N, 1))
        thres_test = 20

        common_shape_np = np.array([nb_rois*N*N, 1])
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
        nb_rois = 5
        nb_gt = 7
        gt_pixels_placeholder_test = np.empty((nb_gt, 3))
        proposals_test = np.ones((nb_rois*N*N, 2))
        #rois_test = np.ones((nb_rois, 2))
        rois_test = None

        gt_pixels_np = gt_pixels_placeholder_test[:,:-1] # shape (nb_gt, 2)
        gt_pixels_np = gt_pixels_np[np.newaxis, :] # shape (1, nb_gt, 2)
        print(gt_pixels_np.shape)
        if rois_test is None:
            gt_pixels_np /= 32.0
            all_gt_pixels_np = np.tile(gt_pixels_np, [proposals_test.shape[0], 1, 1]) # shape (nb_rois*N*N, N*N, 1)
            print(all_gt_pixels_np.shape)
        #else:
        #    gt_pixels_np = gt_pixels_np[np.newaxis, :]
        #    gt_pixels_np = np.tile(gt_pixels_np, [nb_rois*N*N, nb_rois, 1, 1])
        #    broadcast_rois_np = np.expand_dims(np.expand_dims(rois, axis=1), axis=1)
        #    braodcast_rois_np = np.tile(broadcast_rois, [1,


if __name__ == '__main__':
    unittest.main()
