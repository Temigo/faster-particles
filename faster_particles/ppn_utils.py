# *-* encoding: utf-8 *-*
# Utils functions for PPN

import numpy as np
import tensorflow as tf

def generate_anchors(im_shape, repeat=1):
    """
    Generate anchors = centers of pixels.
    Repeat ~ batch size.
    """
    with tf.variable_scope("generate_anchors"):
        dim = len(im_shape) # 2D or 3D
        anchors = np.indices(im_shape).transpose(tuple(range(1, dim+1)) + (0,))
        anchors = anchors + 0.5
        anchors = tf.reshape(tf.constant(anchors, dtype=tf.float32), (-1, dim))
        return tf.tile(anchors, tf.stack([repeat, 1]), name="anchors")

def clip_pixels(pixels, im_shape):
    """
    pixels shape: [None, 2]
    Clip pixels (x, y) to [0, im_shape[0]) x [0, im_shape[1])
    """
    with tf.variable_scope("clip_pixels"):
        dim = len(im_shape) # 2D or 3D
        pixels_final = []
        for i in range(dim):
            pixels_dim = tf.slice(pixels, [0, i], [-1, 1])
            pixels_final.append(tf.clip_by_value(pixels_dim, 0, im_shape[i]))
        pixels = tf.concat(pixels_final, axis=1)
        return pixels

def pixels_transform_inv(pixels, deltas):
    # Given an anchor pixel and regression deltas, estimate proposal pixel
    pred_pixels = pixels + deltas
    return pred_pixels

def top_R_pixels(proposals, scores, R=20, threshold=0.5):
    """
    Order by score and take the top R proposals above threshold.
    Shapes are [N*N, 2] and [N*N, 1]
    """
    with tf.variable_scope("top_R_pixels"):
        # Select top R pixel proposals
        flat_scores = tf.squeeze(scores) # shape N*N
        R = min(R, flat_scores.get_shape().as_list()[0])
        # Output of tf.nn.top_k will be sorted in descending order
        scores, keep = tf.nn.top_k(tf.squeeze(scores), k=R, sorted=True)
        proposals = tf.gather(proposals, keep)
        assert scores.get_shape().as_list() == [R]
        assert keep.get_shape().as_list() == [R]
        # Select scores above threshold
        keep2 = tf.where(tf.greater(scores, threshold))
        assert keep2.get_shape().as_list() == [None, 1]
        proposals_final = tf.gather(proposals, tf.reshape(keep2, (-1,)))
        scores_final = tf.gather(scores, keep2)

        proposals, scores = tf.cond(tf.equal(tf.shape(proposals_final)[0], tf.constant(0)), true_fn=lambda:(tf.slice(proposals, [0, 0], [1, -1]), tf.slice(scores, [0], [1])), false_fn=lambda:(proposals_final, scores_final))
        #assert proposals.get_shape().as_list() == [None, 2]
        return proposals, scores

def predicted_pixels(rpn_cls_prob, rpn_bbox_pred, anchors, im_shape):
    """
    rpn_cls_prob.shape = [None, N, N, n] where n = 2 (background/signal) or num_classes
    rpn_bbox_pred.shape = [None, N, N, 2]
    anchors.shape = [N*N, 2]
    im_shape = (width, height) to clip coordinates of proposals
    Derive predicted pixels from predicted parameters (rpn_bbox_pred) with respect
    to the anchors (= centers of the pixels of the feature map).
    Return a list of predicted pixels and corresponding scores
    of shape [N*N, 2] and [N*N, n]
    """
    with tf.variable_scope("predicted_pixels"):
        # Select pixels that contain something
        scores = rpn_cls_prob[..., 1:]
        scores = tf.reshape(scores, (-1, scores.get_shape().as_list()[-1]))
         # has shape (None, N, N, num_classes - 1)
        dim = anchors.get_shape().as_list()[-1]
        N = rpn_cls_prob.get_shape().as_list()[1]
        # Get proposal pixels from regression deltas of rpn_bbox_pred
        #proposals = pixels_transform_inv(anchors, rpn_bbox_pred)
        anchors = tf.reshape(anchors, shape=tf.stack([-1] + [N] * dim + [dim]))
        proposals =  anchors + rpn_bbox_pred
        proposals = tf.reshape(proposals, (-1, dim))
        # clip predicted pixels to the image
        proposals = clip_pixels(proposals, im_shape)
        rois = tf.cast(proposals, tf.float32, name="rois")
        return rois, scores

def all_combinations(indices):
    return np.array(np.meshgrid(*indices)).T.reshape(-1, len(indices))

def slice_rois(rois, dim2):
    """
    rois shape = None, dim
    Transform ROI (1 pixel on F5) into 4x4 ROIs on F3 (using F5 coordinates)
    """
    with tf.variable_scope("slice_rois"):
        dim = rois.get_shape().as_list()[-1] # 2D or 3D
        rois_slice = [] # Shape dim x nb rois x 1
        for i in range(dim):
            rois_slice.append(tf.slice(rois, [0, i], [-1, 1], name="rois_%d" % dim) * dim2)
        rois_slice = tf.expand_dims(rois_slice, -1) # shape dim x nb rois x 1 x 1
        # FIXME construct rois_slice directly without slicing?
        indices = ([-2, -1, 0, 1],) * dim
        shifts = all_combinations(indices).T[:, np.newaxis, np.newaxis, :] # shape dim x 1 x 1 x nb comb
        all_rois = rois_slice + shifts # using broadcasting => shape dim x nb rois x 1 x nb comb
        #rois = tf.transpose(tf.squeeze(tf.concat(tf.concat(all_rois, axis=1), axis=3)))
        rois = tf.reshape(tf.transpose(all_rois), (-1, dim)) # FIXME do we need to transpose?
        rois = tf.identity(rois / dim2, name="sliced_rois") # (shape nb rois * nb comb) x dim
        return rois

def include_gt_pixels(rois, gt_pixels, dim1, dim2):
    """
    Rois: [None, 2] in F5 coordinates (floating point)
    These ROIs are 4x4 on F3 feature map. Include 3x3 F3 pixels around pixels
    containing ground truth points.
    gt_pixels: shape (None, 2)
    Return rois in F5 coordinates (round coordinates for rois, float for gt rois)
    """
    with tf.variable_scope("include_gt_pixels"):
        dim = gt_pixels.get_shape().as_list()[-1] # 2D or 3D
        # convert to F3 coordinates
        gt_pixels_coord = tf.cast(tf.floor(gt_pixels / dim1), tf.float32)
        # Get 3x3 pixels around this in F3
        gt_pixels_coord = tf.expand_dims(gt_pixels_coord, axis=1)
        #gt_pixels_coord = tf.transpose(gt_pixels_coord, perms=[0, 2, 1])
        gt_pixels_coord = tf.tile(gt_pixels_coord, [1, 3**dim, 1]) # shape N x 9 x 2
        # FIXME clip to image
        shifts = all_combinations(([-1, 0, 1],) * dim)
        update = tf.constant(shifts, dtype=tf.float32)
        update = tf.tile(tf.expand_dims(update, axis=0), [tf.shape(gt_pixels_coord)[0], 1, 1])
        gt_pixels_coord = gt_pixels_coord + update
        gt_pixels_coord = tf.reshape(gt_pixels_coord, (-1, dim)) # Shape N*9, 2
        # FIXME Clip it to F3 size
        # indices = tf.where(tf.less(gt_pixels_coord, 64))
        # gt_pixels_coord = tf.gather_nd(gt_pixels_coord, indices)
        # Go back to F5 coordinates
        gt_pixels_coord = gt_pixels_coord / dim2
        # FIXME As soon as new version of Tensorflow supporting axis option
        # for tf.unique, replace the following rough patch.
        # In the meantime, we will have some duplicates between rois and gt_pixels.
        rois = tf.concat([rois, gt_pixels_coord], axis=0, name="rois") # shape [None, 2]
        assert rois.get_shape().as_list()[-1] == dim and len(rois.get_shape().as_list()) == 2 # Shape [None, 2]
        return rois

def compute_positives_ppn1(gt_pixels, N3, dim1, dim2):
    """
    Returns a mask corresponding to proposals shape = [N*N, 2]
    Positive = 1 = contains a ground truth pixel
    gt_pixels is shape [None, 2]
    Returns classes with shape (16*16,1)
    """
    with tf.variable_scope("ppn1_compute_positives"):
        dim = gt_pixels.get_shape().as_list()[-1]
        classes = tf.zeros(shape=(N3,)*dim)
        # Convert to F5 coordinates (16x16)
        # Shape = None, 2
        gt_pixels = tf.cast(tf.floor(gt_pixels / (dim1 * dim2)), tf.int32)
        # Assign positive pixels based on gt_pixels
        #classes = classes + tf.scatter_nd(gt_pixels, tf.constant(value=1.0, shape=tf.shape(gt_pixels)[0]), classes.shape)
        classes = classes + tf.scatter_nd(gt_pixels, tf.fill((tf.shape(gt_pixels)[0],), 1.0), classes.shape)
        classes = tf.cast(tf.reshape(classes, shape=(-1, 1)), tf.int32)
        classes_mask = tf.cast(classes, tf.bool, name="ppn1_mask") # Turn classes into a mask
        return classes_mask

def compute_positives_ppn2(closest_gt_distance, threshold=2):
    """
    closest_gt_distance shape = (A*N*N, 1)
    Return boolean mask for positives among proposals.
    Positives are those within certain distance range from the
    closest ground-truth point (of the same class? not for now)
    """
    with tf.variable_scope("ppn2_compute_positives"):
        pixel_count = tf.shape(closest_gt_distance)[0]
        common_shape = tf.stack([pixel_count, 1])
        mask = tf.where(tf.greater(closest_gt_distance, threshold), tf.fill(common_shape, 0), tf.fill(common_shape, 1))
        mask = mask + tf.scatter_nd([tf.argmin(closest_gt_distance, output_type=tf.int32)], tf.constant([[1]]), common_shape)
        mask = tf.cast(mask, tf.bool, name="ppn2_mask")
        return mask

def assign_gt_pixels(gt_pixels_placeholder, proposals, dim1, dim2, rois=None):
    """
    Proposals shape: [A*N*N, 2] (N=16 or 64)
    gt_pixels_placeholder is shape [None, 2+1]
    Returns closest ground truth pixels for all pixels and corresponding distance
    -  closest_gt = index of closest gt pixel (of same class)
    - closest_gt_distance = index of closest gt pixel (of same class)
    - closest_gt_label = label of closest gt pixel (regardless of class)
    """
    with tf.variable_scope("assign_gt_pixels"):
        dim = proposals.get_shape().as_list()[-1]
        gt_pixels = tf.slice(gt_pixels_placeholder, [0, 0], [-1, dim])
        gt_pixels = tf.expand_dims(gt_pixels, axis=0)
        # convert proposals to real image coordinates in order to compare with
        # ground truth pixels coordinates
        if rois is not None: # means PPN2
            # Convert to F3 coordinates
            proposals = (proposals + dim2 * rois) * dim1
        else:
            # Convert to F5 coordinates
            proposals = proposals * dim1 * dim2

        # Tile to have shape (A*N*N, None, 2)
        all_gt_pixels = tf.tile(gt_pixels, tf.stack([tf.shape(proposals)[0], 1, 1]))
        all_gt_pixels_mask = tf.fill(tf.shape(all_gt_pixels)[0:2], True)
        # assert all_gt_pixels.get_shape().as_list() == [None, None, dim]
        # Reshape proposals to [A*N*N, 1, 2]
        proposals = tf.expand_dims(proposals, axis=1)
        distances = tf.sqrt(tf.reduce_sum(tf.pow(proposals - all_gt_pixels, 2), axis=2))
        # distances.shape = [A*N*N, None]
        #if rois is not None:
        #   distances = distances + tf.scatter_nd(tf.cast(tf.where(all_gt_pixels_mask), tf.int32), tf.fill((tf.shape(tf.where(all_gt_pixels_mask))[0],), 10000.0), tf.shape(all_gt_pixels_mask))

        # closest_gt.shape = [A*N*N,]
        # closest_gt[i] = indice of closest gt in gt_pixels_placeholder
        closest_gt = tf.argmin(distances, axis=1)
        closest_gt_distance = tf.reduce_min(distances, axis=1, keep_dims=True, name="closest_gt_distance")
        #print("squeezed gt_pixels_placeholder shape=", tf.squeeze(tf.slice(gt_pixels_placeholder, [0,0,0], [-1,1,-1]), axis=1).shape)
        #closest_gt_label = tf.nn.embedding_lookup(tf.slice(gt_pixels_placeholder, [0, 2], [-1, 1]), closest_gt)
        gt_pixels_labels = tf.slice(gt_pixels_placeholder, [0, dim], [-1, 1])
        closest_gt_label = tf.gather_nd(gt_pixels_labels, tf.concat([tf.reshape(tf.range(0, tf.shape(closest_gt_distance)[0]), (-1,1)), tf.cast(tf.reshape(closest_gt, (-1, 1)), tf.int32)], axis=1), name="closest_gt_label")
        return closest_gt, closest_gt_distance, tf.reshape(closest_gt_label, (-1, 1))

def crop_pool_layer(net, rois, dim2, dim):
    """
    Crop and pool intermediate F3 layer.
    Net.shape = [1, 64, 64, 256]
    Rois.shape = [None, 2] # Could be less than R, assumes coordinates on F5
    Also assumes ROIs are 1x1 pixels on F3
    """
    with tf.variable_scope("crop_pool_layer"):
        # Convert rois from F5 coordinates to F3 coordinates (x4)
        rois = tf.cast(rois * dim2, tf.int32) # FIXME 3x3 float coordinates for gt pixel inducted positives
        #print("rois before gather_nd", rois.get_shape().as_list())
        #print("net", net.get_shape().as_list())
        nb_channels = net.get_shape().as_list()[-1]
        indices = tf.concat([tf.fill([tf.shape(rois)[0], 1], 0), rois], axis=1)
        #print(indices.get_shape().as_list())
        rois = tf.gather_nd(net, indices, name="crop_layer")
        #print(rois.get_shape().as_list())
        return tf.reshape(rois, (-1,) + (1,) * dim + (nb_channels,))
