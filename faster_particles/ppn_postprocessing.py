# *-* encoding: utf-8 *-*
# DBSCAN and NMS postprocessing for PPN

import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN

def filter_points(im_proposals, im_scores, eps):
    """
    DBSCAN postprocessing on point proposals.
    """
    db = DBSCAN(eps=eps, min_samples=1).fit_predict(im_proposals)
    keep = {}
    index = {}
    new_proposals = []
    new_scores = []
    clusters, index = np.unique(db, return_index=True)
    for i in clusters:
        indices = np.where(db == i)
        new_proposals.append(np.average(im_proposals[indices], axis=0)) # weights=im_scores[indices]
        new_scores.append(np.average(im_scores[indices], axis=0))
    return np.array(new_proposals), np.array(new_scores), index

def nms_step(order, areas, proposals, new_proposals, keep, threshold, size, *args):
    """
    A single NMS step. See nms function for more details.
    """
    i = order[0]
    keep = tf.concat([keep, [i]], axis=0)
    dim = len(args)/2
    inter = tf.ones((tf.shape(order)[0]-1,)) # area/volume of intersection
    proposals_inside = proposals
    for d in range(dim):
        xx1 = tf.maximum(args[d][i], tf.gather(args[d], order[1:]))
        xx2 = tf.minimum(args[dim+d][i], tf.gather(args[dim+d], order[1:]))
        inter = inter * tf.maximum(0.0, xx2 - xx1 + 1)
        indices_inside = tf.where(tf.logical_and(proposals_inside[:, d] >= args[d][i], proposals_inside[:, d] <= args[dim+d][i]))
        proposals_inside = tf.gather_nd(proposals_inside, indices_inside)

    # Compute IoU
    ovr = inter / (tf.gather(areas, i) + tf.gather(areas, order[1:]) - inter)
    indices = tf.where(ovr <= threshold)
    new_order = tf.gather(order, indices + 1)[:, 0]
    current_coord = tf.reduce_mean(proposals_inside, axis=0)
    #current_coord = proposals[i]
    new_proposals = tf.concat([new_proposals[:i], [current_coord], new_proposals[i+1:]], axis=0)
    return (new_order, areas, proposals, new_proposals, keep, threshold, size) + args

def nms(im_proposals, im_scores, threshold=0.01, size=6.0):
    """
    Performs NMS (non maximal suppression) postprocessing on proposed pixels.
    - Look at pixels in order of decreasing score
    - Consider squares of size `size` centered at each pixels
    - If the IoU is bigger than a threshold don't keep the point
    """
    size = size
    areas = tf.ones((tf.shape(im_proposals)[0],))
    coords = ()
    dim = im_proposals.get_shape()[-1]
    for d in range(dim):
        coords = coords + (im_proposals[:, d] - size,)
    for d in range(dim):
        coords = coords + (im_proposals[:, d] + size,)
    for d in range(dim):
        areas = areas * (coords[dim+d] - coords[d] + 1.0)
    coords_shape = ()
    for c in coords:
        coords_shape = coords_shape + (c.get_shape(),)
    _, order = tf.nn.top_k(im_scores, k=tf.shape(im_scores)[0])
    keep = tf.Variable([0], dtype=tf.int32)
    threshold = tf.constant(threshold)
    size = tf.constant(size)
    #new_proposals = tf.Variable(im_proposals, validate_shape=False)
    while_return = tf.while_loop(lambda order, *args: tf.shape(order)[0] > 0, nms_step, [order, areas, im_proposals, im_proposals, keep, threshold, size] + list(coords), shape_invariants=[order.get_shape(), areas.get_shape(), im_proposals.get_shape(), im_proposals.get_shape(), tf.TensorShape((None,)), threshold.get_shape(), size.get_shape()] + list(coords_shape))
    keep = while_return[4][1:]
    new_proposals = while_return[3]
    return new_proposals, keep
