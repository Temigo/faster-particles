# *-* encoding: utf-8 *-*
# Utils functions for PPN

import numpy as np
import tensorflow as tf

def generate_anchors(width, height, repeat=1):
    """
    Generate anchors = centers of pixels.
    Repeat ~ batch size.
    """
    with tf.variable_scope("generate_anchors"):
        anchors = np.indices((width, height)).transpose((1, 2, 0))
        anchors = anchors + 0.5
        anchors = tf.reshape(tf.constant(anchors, dtype=tf.float32), (-1, 2))
        return tf.tile(anchors, tf.stack([repeat, 1]), name="anchors")

def clip_pixels(pixels, im_shape):
    """
    pixels shape: [None, 2]
    Clip pixels (x, y) to [0, im_shape[0]) x [0, im_shape[1])
    """
    with tf.variable_scope("clip_pixels"):
        pixels_x = tf.slice(pixels, [0, 0], [-1, 1])
        pixels_y = tf.slice(pixels, [0, 1], [-1, 1])
        pixels_x = tf.clip_by_value(pixels_x, 0, im_shape[0])
        pixels_y = tf.clip_by_value(pixels_y, 0, im_shape[1])
        pixels = tf.concat([pixels_x, pixels_y], axis=1)
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

def predicted_pixels(rpn_cls_prob, rpn_bbox_pred, anchors, im_shape, R=20, classes=False):
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
        #if not classes:
        scores = rpn_cls_prob[:, :, :, 1:]
        #scores = tf.reshape(scores, (-1, rpn_cls_prob.get_shape().as_list()[-1]-1))
        #else:
        #    scores = rpn_cls_prob
        scores = tf.reshape(scores, (-1, scores.get_shape().as_list()[-1]))
         # has shape (None, N, N, num_classes - 1)

        # Get proposal pixels from regression deltas of rpn_bbox_pred
        #proposals = pixels_transform_inv(anchors, rpn_bbox_pred)
        anchors = tf.reshape(anchors, shape=(-1, rpn_cls_prob.get_shape().as_list()[1], rpn_cls_prob.get_shape().as_list()[2], 2))
        proposals =  anchors + rpn_bbox_pred
        proposals = tf.reshape(proposals, (-1, 2))
        # clip predicted pixels to the image
        proposals = clip_pixels(proposals, im_shape)
        rois = tf.cast(proposals, tf.float32, name="rois")
        return rois, scores

def slice_rois(rois, dim2):
    """
    Transform ROI (1 pixel on F5) into 4x4 ROIs on F3 (using F5 coordinates)
    """
    with tf.variable_scope("slice_rois"):
        rois_x = tf.slice(rois, [0, 0], [-1, 1], name="rois_x") * dim2
        rois_y = tf.slice(rois, [0, 1], [-1, 1], name="rois_y") * dim2
        rois = tf.concat([
            tf.concat([rois_x, rois_y], axis=1),
            tf.concat([rois_x, rois_y+1], axis=1),
            tf.concat([rois_x, rois_y-1], axis=1),
            tf.concat([rois_x, rois_y-2], axis=1),
            tf.concat([rois_x+1, rois_y], axis=1),
            tf.concat([rois_x+1, rois_y+1], axis=1),
            tf.concat([rois_x+1, rois_y-1], axis=1),
            tf.concat([rois_x+1, rois_y-2], axis=1),
            tf.concat([rois_x-1, rois_y], axis=1),
            tf.concat([rois_x-1, rois_y+1], axis=1),
            tf.concat([rois_x-1, rois_y-1], axis=1),
            tf.concat([rois_x-1, rois_y-2], axis=1),
            tf.concat([rois_x-2, rois_y], axis=1),
            tf.concat([rois_x-2, rois_y+1], axis=1),
            tf.concat([rois_x-2, rois_y-1], axis=1),
            tf.concat([rois_x-2, rois_y-2], axis=1),
        ], axis=0, name="sliced_rois")
        rois = rois / dim2
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
        # convert to F3 coordinates
        gt_pixels_coord = tf.cast(tf.floor(gt_pixels / dim1), tf.float32)
        # Get 3x3 pixels around this in F3
        gt_pixels_coord = tf.expand_dims(gt_pixels_coord, axis=1)
        #gt_pixels_coord = tf.transpose(gt_pixels_coord, perms=[0, 2, 1])
        gt_pixels_coord = tf.tile(gt_pixels_coord, [1, 9, 1]) # shape N x 9 x 2
        # FIXME clip to image
        update = tf.constant([[0, 0], [0, 1], [0, -1], [1, 0], [1, 1], [1, -1], [-1, 0], [-1, 1], [-1, -1]], dtype=tf.float32)
        update = tf.tile(tf.expand_dims(update, axis=0), [tf.shape(gt_pixels_coord)[0], 1, 1])
        gt_pixels_coord = gt_pixels_coord + update
        gt_pixels_coord = tf.reshape(gt_pixels_coord, (-1, 2)) # Shape N*9, 2
        # FIXME Clip it to F3 size
        # indices = tf.where(tf.less(gt_pixels_coord, 64))
        # gt_pixels_coord = tf.gather_nd(gt_pixels_coord, indices)
        # Go back to F5 coordinates
        gt_pixels_coord = gt_pixels_coord / dim2
        # FIXME As soon as new version of Tensorflow supporting axis option
        # for tf.unique, replace the following rough patch.
        # In the meantime, we will have some duplicates between rois and gt_pixels.
        rois = tf.concat([rois, gt_pixels_coord], axis=0, name="rois") # shape [None, 2]
        assert rois.get_shape().as_list()[-1] == 2 and len(rois.get_shape().as_list()) == 2 # Shape [None, 2]
        return rois

def compute_positives_ppn1(gt_pixels, N3, dim1, dim2):
    """
    Returns a mask corresponding to proposals shape = [N*N, 2]
    Positive = 1 = contains a ground truth pixel
    gt_pixels is shape [None, 2]
    Returns classes with shape (16*16,1)
    """
    with tf.variable_scope("ppn1_compute_positives"):
        classes = tf.zeros(shape=(N3, N3)) # FIXME don't hardcode 16
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
        gt_pixels = tf.slice(gt_pixels_placeholder, [0, 0], [-1, 2])
        gt_pixels = tf.expand_dims(gt_pixels, axis=0)
        # convert proposals to real image coordinates in order to compare with
        # ground truth pixels coordinates
        if rois is not None: # means PPN2
            #gt_pixels = gt_pixels / dim1 # Convert to F3 coordinates
            proposals = (proposals + dim2 * rois) * dim1
        else:
            #gt_pixels = gt_pixels / (dim1 * dim2) # Convert to F5 coordinates
            proposals = proposals * dim1 * dim2

        # Tile to have shape (A*N*N, None, 2)
        all_gt_pixels = tf.tile(gt_pixels, tf.stack([tf.shape(proposals)[0], 1, 1]))
        all_gt_pixels_mask = tf.fill(tf.shape(all_gt_pixels)[0:2], True)

        """else: # Translate each batch of N*N rows of all_gt_pixels w.r.t. corresponding ROI center
            # FIXME check that this yields expected result
            # Translation is gt_pixels / 8.0 - 4*rois[i] (with conversion to F3 coordinates)
            # Go to shape [1, 1, None, 2]
            gt_pixels = tf.expand_dims(gt_pixels, axis=0)
            # Tile to shape [A, N*N, None, 2]
            gt_pixels = tf.tile(gt_pixels, [tf.shape(rois)[0], tf.cast(tf.shape(proposals)[0]/tf.shape(rois)[0], tf.int32), 1, 1])
            # Broadcast translation
            broadcast_rois = tf.expand_dims(tf.expand_dims(rois, axis=1), axis=1)
            broadcast_rois = tf.tile(broadcast_rois, [1, tf.shape(gt_pixels)[1], tf.shape(gt_pixels)[2], 1])
            all_gt_pixels = gt_pixels / 8.0 - 4.0 * broadcast_rois
            # Reshape to [A*N*N, None, 2]
            all_gt_pixels = tf.reshape(all_gt_pixels, (tf.shape(proposals)[0], -1, 2))
            gt_pixels_labels = tf.tile(tf.expand_dims(tf.reshape(gt_pixels_placeholder[:, -1], (tf.shape(gt_pixels_placeholder)[0], 1)), axis=0), (tf.shape(scores)[0], 1, 1))
            # Reshape scores to A*N*N, num_classes
            scores = tf.reshape(scores, (-1, tf.shape(scores)[-1]))
            print(tf.expand_dims(tf.expand_dims(tf.argmax(scores, axis=1), axis=1), axis=2))
            scores_labels = tf.cast(tf.tile(tf.expand_dims(tf.expand_dims(tf.argmax(scores, axis=1), axis=1), axis=2), [1, tf.shape(gt_pixels_placeholder)[0], 1]), dtype=tf.float32)
            all_gt_pixels_mask = tf.where(tf.equal(gt_pixels_labels, scores_labels), tf.fill(tf.shape(scores_labels), False), tf.fill(tf.shape(scores_labels), True))
            all_gt_pixels_mask = tf.squeeze(all_gt_pixels_mask, axis=2)
        """
        assert all_gt_pixels.get_shape().as_list() == [None, None, 2]
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
        gt_pixels_labels = tf.slice(gt_pixels_placeholder, [0, 2], [-1, 1])
        closest_gt_label = tf.gather_nd(gt_pixels_labels, tf.concat([tf.reshape(tf.range(0, tf.shape(closest_gt_distance)[0]), (-1,1)), tf.cast(tf.reshape(closest_gt, (-1, 1)), tf.int32)], axis=1), name="closest_gt_label")
        print(closest_gt_label.get_shape().as_list())
        return closest_gt, closest_gt_distance, tf.reshape(closest_gt_label, (-1, 1))
