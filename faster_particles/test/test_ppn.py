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
    include_gt_pixels, predicted_pixels, crop_pool_layer, all_combinations, slice_rois
from faster_particles.toydata.toydata_generator import ToydataGenerator

def generate_anchors_np(im_shape, repeat=1):
    dim = len(im_shape)
    anchors = np.indices(im_shape).transpose(tuple(range(1, dim+1)) + (0,))
    anchors = anchors + 0.5
    anchors = np.reshape(anchors, (-1, dim))
    return np.repeat(anchors, repeat, axis=0)

def clip_pixels_np(pixels, im_shape):
    """
    pixels shape: [None, 2]
    Clip pixels (x, y) to [0, im_shape[0]) x [0, im_shape[1])
    """
    dim = len(im_shape)
    for i in range(dim):
        pixels[:, i] = np.clip(pixels[:, i], 0, im_shape[i])
    return pixels

class Test(unittest.TestCase):
    #self.toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)
    #self.net = PPN()
    def generate_anchors(self, im_shape, repeat):
        anchors_np = generate_anchors_np(im_shape, repeat=repeat)
        with tf.Session() as sess:
            anchors_tf = generate_anchors(im_shape, repeat=repeat)
            return np.array_equal(anchors_tf, anchors_np)

    def test_generate_anchors_2d(self):
        im_shape = (2, 2)
        repeat = 3
        return self.generate_anchors(im_shape, repeat)

    def test_generate_anchors_3d(self):
        im_shape = (2, 2, 2)
        repeat = 3
        return self.generate_anchors(im_shape, repeat)

    def clip_pixels(self, im_shape, proposals_np):
        pixels_np = clip_pixels_np(proposals_np, im_shape)
        with tf.Session() as sess:
            proposals = tf.constant(proposals_np, dtype=tf.float32)
            pixels = clip_pixels(proposals, im_shape)
            pixels_tf = sess.run(pixels)
            return np.allclose(pixels_np, pixels_tf)

    def test_clip_pixels_2d(self):
        im_shape = (3, 3)
        proposals_np = np.array([[-0.5, 1.0], [0.01, 3.4], [2.5, 2.99]])
        return self.clip_pixels(im_shape, proposals_np)

    def test_clip_pixels_3d(self):
        im_shape = (2, 2, 2)
        proposals_np = np.random.rand(5, 3)*4-1
        return self.clip_pixels(im_shape, proposals_np)

    def top_R_pixels(self, R, threshold, proposals_np, scores_np):
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

    def test_top_R_pixels_2d(self):
        R = 3
        threshold = 0.5
        # Shape N*N x 2
        proposals_np = np.array([[0.0, 1.0], [0.5, 0.7], [0.3, 0.88], [-0.2, 0.76], [0.23, 0.47], [0.33, 0.56], [0.0, 0.4], [-0.6, 0.3], [0.27, -0.98]])
        # Shape N*N x 1
        scores_np = np.array([0.1, 0.5, 0.7, 0.45, 0.65, 0.01, 0.78, 0.98, 0.72])
        return self.top_R_pixels(R, threshold, proposals_np, scores_np)

    def test_top_R_pixels_3d(self):
        R = 3
        threshold = 0.5
        # shape N*N x 3
        proposals_np = np.array([[0.0, 1.0, 0.3], [0.87, 0.1, -0.34], [0.45, 0.68, 0.09],
                                [0.34, 0.21, -0.6], [0.12, -0.4, 0.8], [0.48, 0.43, -0.79], [0.89, 0.05, -0.02], [0.9, 0.04, 1.0]])
        # shape N*N x 1
        scores_np = np.array([0.1, 0.5, 0.7, 0.45, 0.65, 0.01, 0.78, 0.98])
        return self.top_R_pixels(R, threshold, proposals_np, scores_np)

    def predicted_pixels(self, im_shape, repeat, rpn_cls_prob_np, rpn_bbox_pred_np):
        dim = len(im_shape)
        anchors_np = generate_anchors_np(im_shape, repeat=repeat)
        scores = rpn_cls_prob_np[..., 1:]
        roi_scores_np = np.reshape(scores, (-1, scores.shape[-1]))
        anchors_np = np.reshape(anchors_np, (-1,) + (rpn_cls_prob_np.shape[1],) * dim + (dim,))
        proposals =  anchors_np + rpn_bbox_pred_np
        proposals = np.reshape(proposals, (-1, dim))
        # clip predicted pixels to the image
        proposals = clip_pixels_np(proposals, im_shape) # FIXME np function
        rois_np = proposals.astype(float)

        with tf.Session() as sess:
            anchors_tf = generate_anchors(im_shape, repeat=repeat)
            rpn_cls_prob_tf = tf.constant(rpn_cls_prob_np, dtype=tf.float32)
            rpn_bbox_pred_tf = tf.constant(rpn_bbox_pred_np, dtype=tf.float32)
            rois, roi_scores = predicted_pixels(rpn_cls_prob_tf, rpn_bbox_pred_tf, anchors_tf, im_shape)
            rois_tf, roi_scores_tf = sess.run([rois, roi_scores])
            return np.allclose(rois_tf, rois_np) and np.allclose(roi_scores_tf, roi_scores_np)

    def test_predicted_pixels1_2d(self): # for PPN1
        im_shape = (2, 2)
        repeat = 1
        # Shape [None, N, N, n] where n = 2 (background/signal)
        rpn_cls_prob_np = np.array([[[[0.1, 0.9], [0.3, 0.7]], [[0.5, 0.5], [0.8, 0.2]]]])
        # Shape [None, N, N, 2]
        rpn_bbox_pred_np = np.array([[[[0.1, 0.1], [0.5, 0.2]], [[0.9, -0.5], [0.1, -0.4]]]])
        return self.predicted_pixels(im_shape, repeat, rpn_cls_prob_np, rpn_bbox_pred_np)

    def test_predicted_pixels1_3d(self):
        im_shape = (2, 2, 2)
        repeat = 1
        rpn_cls_prob_np = np.random.rand(1, 2, 2, 2, 2)
        rpn_bbox_pred_np = np.random.rand(1, 2, 2, 2, 3)*2-1
        return self.predicted_pixels(im_shape, repeat, rpn_cls_prob_np, rpn_bbox_pred_np)

    def test_predicted_pixels2_2d(self): # for PPN2
        im_shape = (2, 2)
        repeat = 1
        # Shape [None, N, N, n] where n = num_classes
        rpn_cls_prob_np = np.array([[[[0.1, 0.8, 0.1], [0.3, 0.65, 0.05]], [[0.5, 0.02, 0.48], [0.8, 0.18, 0.02]]]])
        # Shape [None, N, N, 2]
        rpn_bbox_pred_np = np.array([[[[0.1, 0.1], [0.5, 0.2]], [[0.9, -0.5], [0.1, -0.4]]]])
        return self.predicted_pixels(im_shape, repeat, rpn_cls_prob_np, rpn_bbox_pred_np)

    def test_predicted_pixels2_3d(self):
        im_shape = (2, 2, 2)
        repeat = 1
        rpn_cls_prob_np = np.random.rand(1, 2, 2, 2, 3)
        rpn_bbox_pred_np = np.random.rand(1, 2, 2, 2, 3)*2-1
        return self.predicted_pixels(im_shape, repeat, rpn_cls_prob_np, rpn_bbox_pred_np)

    def include_gt_pixels(self, rois_np, gt_pixels_np, dim1, dim2):
        dim = gt_pixels_np.shape[-1]
        # convert to F3 coordinates
        gt_pixels_coord = np.floor(gt_pixels_np / dim1)
        # Get 3x3 pixels around this in F3
        gt_pixels_coord = gt_pixels_coord[:, np.newaxis, :]
        gt_pixels_coord = np.tile(gt_pixels_coord, [1, 3**dim, 1]) # shape N x 9 x 2
        shifts = all_combinations(([-1, 0, 1],) * dim)
        update = np.tile(shifts[np.newaxis, :, :], [gt_pixels_coord.shape[0], 1, 1])
        gt_pixels_coord = gt_pixels_coord + update
        gt_pixels_coord = np.reshape(gt_pixels_coord, (-1, dim)) # Shape N*9, 2
        # Go back to F5 coordinates
        gt_pixels_coord = gt_pixels_coord / dim2
        rois_result_np = np.vstack([np.floor(rois_np), gt_pixels_coord]) # shape [None, 2]

        with tf.Session() as sess:
            rois_tf = tf.constant(rois_np, dtype=tf.float32)
            gt_pixels_tf = tf.constant(gt_pixels_np, dtype=tf.float32)
            rois_tf = include_gt_pixels(rois_tf, gt_pixels_tf, dim1, dim2)
            rois_result_tf = sess.run(rois_tf)
            return np.allclose(rois_result_tf, rois_result_np)

    def test_include_gt_pixels_2d(self):
        dim1, dim2 = 8.0, 4.0
        # [None, 2] in F5 coordinates
        rois_np = np.array([[0, 3], [15, 2], [3, 4], [5.6, 9.1]])
        # [None, 2]
        gt_pixels_np = np.array([[2.4, 2.3], [3, 4], [6.4, 1.2]])
        return self.include_gt_pixels(rois_np, gt_pixels_np, dim1, dim2)

    def test_include_gt_pixels_3d(self):
        dim1, dim2 = 8.0, 4.0
        rois_np = np.random.rand(10, 3)
        gt_pixels_np = np.random.rand(4, 3)*dim1*dim2
        return self.include_gt_pixels(rois_np, gt_pixels_np, dim1, dim2)

    def compute_positives_ppn1(self, gt_pixels_test, N3, dim1, dim2):
        dim =gt_pixels_test.shape[-1]
        classes_np = np.zeros((N3,)*dim)
        gt_pixels_np = np.floor(gt_pixels_test / (dim1 * dim2)).astype(int)
        gt_pixels_np = tuple(zip(*gt_pixels_np))
        classes_np[gt_pixels_np] = 1.
        classes_mask_np = classes_np.reshape(-1,1).astype(bool) # shape (16*16, 1)

        with tf.Session() as sess:
            gt_pixels_tf = tf.constant(gt_pixels_test, dtype=tf.float32)
            classes_mask_tf = compute_positives_ppn1(gt_pixels_tf, N3, dim1, dim2)
            classes_mask_tf = sess.run([classes_mask_tf])
        return np.allclose(classes_mask_np, classes_mask_tf)

    def test_compute_positives_ppn1_2d(self):
        dim1, dim2, N3 = 8.0, 4.0, 16
        # Dummy input for testing, num of gt pixels = N = 3
        gt_pixels_test = np.array([[5.5, 7.7], [511.1, 433.3], [320, 320]])
        return self.compute_positives_ppn1(gt_pixels_test, N3, dim1, dim2)

    def test_compute_positives_ppn1_3d(self):
        dim1, dim2, N3 = 8.0, 4.0, 16
        gt_pixels_test = np.array([[5.5, 7.7, 45.9], [511.1, 433.3, 5.6], [320, 320, 201]])
        return self.compute_positives_ppn1(gt_pixels_test, N3, dim1, dim2)

    def compute_positives_ppn2(self, closest_gt_distance_test, thres_test):
        pixel_count = closest_gt_distance_test.shape[0]
        common_shape_np = np.array([pixel_count, 1])
        mask_np = np.where(np.greater(closest_gt_distance_test, thres_test), False, True)
        mask_np[np.argmin(closest_gt_distance_test)] = True
        with tf.Session() as sess:
            mask_tf = compute_positives_ppn2(closest_gt_distance_test, threshold=thres_test)
            mask_tf = sess.run([mask_tf])
        return np.allclose(mask_np, mask_tf)

    def test_compute_positives_ppn2_2d(self):
        nb_rois, N = 5, 16
        closest_gt_distance_test = np.arange(nb_rois*N*N).reshape(-1, 1)
        thres_test = 2
        return self.compute_positives_ppn2(closest_gt_distance_test, thres_test)

    def test_compute_positives_ppn2_3d(self):
        nb_rois, N = 5, 16
        closest_gt_distance_test = np.arange(nb_rois*N*N*N).reshape(-1, 1)
        thres_test = 2
        return self.compute_positives_ppn2(closest_gt_distance_test, thres_test)

    # TODO test rois option too
    def assign_gt_pixels(self, gt_pixels_np, proposals_np, dim1, dim2, rois=None):
        dim = proposals_np.shape[-1]
        gt_pixels = gt_pixels_np[:, :-1]
        gt_pixels = gt_pixels[np.newaxis, :, :]
        if rois is not None:
            proposals = (proposals_np * dim2 * rois) * dim1
        else:
            proposals = proposals_np * dim1 * dim2
        all_gt_pixels = np.tile(gt_pixels, [proposals_np.shape[0], 1, 1])
        proposals = proposals[:, np.newaxis, :]
        distances = np.sqrt(np.sum(np.power(proposals - all_gt_pixels, 2), axis=2))
        closest_gt = np.argmin(distances, axis=1)
        closest_gt_distance = np.amin(distances, axis=1)
        gt_pixels_labels = gt_pixels_np[:, -1]
        closest_gt_label = [gt_pixels_labels[i] for i in closest_gt]

        with tf.Session() as sess:
            gt_pixels_tf = tf.constant(gt_pixels_np, dtype=tf.float32)
            proposals_tf = tf.constant(proposals_np, dtype=tf.float32)
            closest_gt_tf, closest_gt_distance_tf, closest_gt_label_tf = assign_gt_pixels(gt_pixels_tf, proposals_tf, dim1, dim2, rois=rois)
            closest_gt_result, closest_gt_distance_result, closest_gt_label_result = sess.run([closest_gt_tf, closest_gt_distance_tf, closest_gt_label_tf])
            return np.allclose(closest_gt_result, closest_gt) and np.allclose(closest_gt_distance_result, closest_gt_distance) and np.allclose(closest_gt_label_result, closest_gt_label)

    def test_assign_gt_pixels_2d(self):
        dim1, dim2 = 8.0, 4.0
        gt_pixels_np = np.array([[0.5, 5.6, 1], [53, 76, 2]])
        proposals_np = np.array([[1.0, 1.0], [7, 75], [98, 10], [5, 34]])
        return self.assign_gt_pixels(gt_pixels_np, proposals_np, dim1, dim2)

    def test_assign_gt_pixels_3d(self):
        dim1, dim2 = 8.0, 4.0
        gt_pixels_np = np.array([[0.5, 5.6, 45, 1], [53, 76, 102, 2]])
        proposals_np = np.array([[1.0, 1.0, 0.43], [7, 75, 2.3], [98, 10, 45], [5, 34, 72]])
        return self.assign_gt_pixels(gt_pixels_np, proposals_np, dim1, dim2)

    def crop_pool_layer(self, net, rois_np, dim2, dim):
        rois = np.array(rois_np * dim2).astype(int)
        nb_channels = net.shape[-1]
        if dim == 2:
            rois = [net[:, i[0], i[1], :] for i in rois]
        elif dim == 3:
            rois = [net[:, i[0], i[1], i[2], :] for i in rois]
        rois = np.reshape(rois, (-1,) + (1,) * dim + (nb_channels,))
        with tf.Session() as sess:
            rois_tf = crop_pool_layer(tf.constant(net, dtype=tf.float32), tf.constant(rois_np, dtype=tf.float32), dim2, dim)
            rois_result = sess.run(rois_tf)
            return np.allclose(rois, rois_result)

    def test_crop_pool_layer_2d(self):
        dim2, dim = 4.0, 2
        net = np.random.rand(1, 64, 64, 16)
        rois_np = np.random.rand(10, 2)*16
        return self.crop_pool_layer(net, rois_np, dim2, dim)

    def test_crop_pool_layer_3d(self):
        dim2, dim = 4.0, 3
        net = np.random.rand(1, 64, 64, 64, 16)
        rois_np = np.random.rand(10, 3)*16
        return self.crop_pool_layer(net, rois_np, dim2, dim)

    def test_all_combinations(self):
        return np.allclose(all_combinations(([0, 1], [0, 1])), np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    def slice_rois(self, rois_np, dim2):
        dim = rois_np.shape[-1]
        rois_slice = []
        for i in range(dim):
            rois_slice.append(np.multiply(rois_np[:, i], dim2))
        rois_slice = np.array(rois_slice)[..., np.newaxis, np.newaxis]
        indices = ([-2, -1, 0, 1],) * dim
        shifts = all_combinations(indices).T[:, np.newaxis, np.newaxis, :]
        all_rois = np.add(rois_slice, shifts)
        rois = np.reshape(all_rois, (-1, dim)) / dim2
        with tf.Session() as sess:
            rois_tf = slice_rois(tf.constant(rois_np, dtype=tf.float32), dim2)
            rois_result = sess.run(rois_tf)
            return np.allclose(rois, rois_result)

    def test_slice_rois_2d(self):
        dim2 = 4.0
        rois_np = np.random.rand(10, 2) * 64
        return self.slice_rois(rois_np, dim2)

    def test_slice_rois_3d(self):
        dim2 = 4.0
        rois_np = np.random.rand(10, 3) * 64
        return self.slice_rois(rois_np, dim2)

if __name__ == '__main__':
    unittest.main()
