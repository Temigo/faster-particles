import tensorflow as tf
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import ctypes
from ctypes import *


def nvtxso_push():
    dll = ctypes.CDLL('./cuda_nvtx.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.nvtx_push
    func.argtypes = [c_char_p]
    return func


def nvtxso_pop():
    dll = ctypes.CDLL('./cuda_nvtx.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.nvtx_pop
    return func


__nvtx_push = nvtxso_push();
__nvtx_pop = nvtxso_pop();


def nms_numpy(im_proposals, im_scores, threshold=0.01, size=6.0):
    x1 = im_proposals[:, 0] - size
    x2 = im_proposals[:, 0] + size
    y1 = im_proposals[:, 1] - size
    y2 = im_proposals[:, 1] + size

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = im_scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return im_proposals[keep], keep


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
    # current_coord = proposals[i]
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
    while_return = tf.while_loop(
        lambda order, *args: tf.shape(order)[0] > 0,
        nms_step,
        [order, areas, im_proposals, im_proposals, keep, threshold, size] + list(coords),
        shape_invariants=[order.get_shape(), areas.get_shape(),
                          im_proposals.get_shape(), im_proposals.get_shape(),
                          tf.TensorShape((None,)), threshold.get_shape(),
                          size.get_shape()] + list(coords_shape))
    keep = while_return[4][1:]
    new_proposals = while_return[3]
    return new_proposals, keep


def nms_test1():
    # N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    __nvtx_push("nvtx-1".encode('utf-8'))
    N_values = [100]
    durations1, durations2 = [], []
    for N in N_values:
        im_proposals = tf.random_uniform((N, 2), maxval=192)
        im_scores = 0.1 * tf.ones((N,))
        _, order = tf.nn.top_k(im_scores, k=tf.shape(im_scores)[0])
        print(order)
        threshold = 0.01
        size = 6.0
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
        print(coords)
        for c in coords:
            coords_shape = coords_shape + (c.get_shape(),)
        keep = tf.Variable([0], dtype=tf.int32)
        threshold = tf.constant(threshold)
        size = tf.constant(size)
        print([order, areas, im_proposals, im_proposals, keep, threshold, size] + list(coords))
        while_return = tf.while_loop(
            lambda order, *args: tf.shape(order)[0] > 0,
            nms_step,
            [order, areas, im_proposals, im_proposals, keep, threshold, size] + list(coords),
            shape_invariants=[order.get_shape(), tf.TensorShape((None,)),
                              tf.TensorShape((None, 2)), tf.TensorShape((None, 2)),
                              tf.TensorShape((None,)), threshold.get_shape(),
                              size.get_shape(), tf.TensorShape((None,)), tf.TensorShape((None,)), tf.TensorShape((None,)), tf.TensorShape((None,))])
        keep = while_return[4][1:]
        new_proposals = while_return[3]

        MAX_STEPS = 100
        duration1, duration2 = 0, 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_STEPS):
                start = time.time()
                sess.run([order], feed_dict={})
                end = time.time()
                duration1 += end - start
                start = time.time()
                sess.run([while_return], feed_dict={})
                end = time.time()
                duration2 += end - start
        duration1 /= MAX_STEPS
        duration2 /= MAX_STEPS
        durations1.append(duration1)
        durations2.append(duration2)
        print("N = %d - Average duration = %f ms + %f ms" % (N, duration1, duration2))

    __nvtx_pop();
    print(np.stack([np.array(N_values), np.array(durations1), np.array(durations2)], axis=1))


def nms_test2():
    # N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    N_values = [100]
    durations = []
    for N in N_values:
        im_proposals = tf.random_uniform((N, 2), maxval=192)
        im_scores = 0.1 * tf.ones((N,))
        new_proposals, keep = tf.py_func(nms_numpy, [im_proposals, im_scores], (tf.float32, tf.int64))

        MAX_STEPS = 100
        duration = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_STEPS):
                start = time.time()
                sess.run([new_proposals], feed_dict={})
                end = time.time()
                duration += end - start
        duration /= MAX_STEPS
        durations.append(duration)
        print("N = %d - Average duration = %f ms" % (N, duration))

    print(np.stack([np.array(N_values), np.array(durations)], axis=1))


def topk_test():
    # N_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 1000]
    # N_values = [1e2, 1e3, 1e4, 1e5]
    N_values = [100]
    durations = []
    __nvtx_push("nvtx-1".encode('utf-8'));
    for N in N_values:
        print(N)
        array = tf.random_uniform((N,), maxval=192)
        topk = tf.nn.top_k(array, k=tf.shape(array)[0])
        duration = 0
        MAX_STEPS = 10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(MAX_STEPS):
                start = time.time()
                sess.run([topk], feed_dict={})
                end = time.time()
                duration += end - start

        duration /= MAX_STEPS
        durations.append(duration)
        print("N = %d - Average duration of inference = %f ms" % (N, duration))

    __nvtx_pop();

    # fig = plt.figure()
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.set_xlim(0, 700)
    # ax.set_ylim(0, 0.01)
    # ax.plot(N_values, durations, 'bo')
    # plt.savefig('display/topk.png', bbox_inches='tight')
    # plt.close(fig)

    print(N_values, durations)
    print(np.stack([np.array(N_values), np.array(durations)], axis=1))

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # topk_test()
    nms_test1()
    # nms_test2()
