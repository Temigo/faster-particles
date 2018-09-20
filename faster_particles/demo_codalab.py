from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import time
import tables

from faster_particles.display_utils import display_uresnet
from faster_particles.base_net import basenets
from faster_particles.metrics import UResNetMetrics
from faster_particles.demo_ppn import get_data, load_weights


def inference(cfg, is_testing=False):
    """
    Inference for either PPN or (xor) base network (e.g. UResNet)
    """
    if not os.path.isdir(cfg.DISPLAY_DIR):
        os.makedirs(cfg.DISPLAY_DIR)

    if is_testing:
        _, data = get_data(cfg)
    else:
        data, _ = get_data(cfg)

    net = basenets[cfg.BASE_NET](cfg=cfg)
    if cfg.WEIGHTS_FILE_PPN is None and cfg.WEIGHTS_FILE_BASE is None:
        raise Exception("Need a checkpoint file")

    net.init_placeholders()
    net.create_architecture(is_training=False)
    duration = 0

    metrics = UResNetMetrics(cfg)
    FILTERS = tables.Filters(complevel=5, complib='zlib', shuffle=True,
                             bitshuffle=False, fletcher32=False,
                             least_significant_digit=None)
    f_submission = tables.open_file('/data/codalab/submission_5-6.hdf5', 'w',
                                    filters=FILTERS)
    preds_array = f_submission.create_earray('/', 'pred', tables.UInt32Atom(),
                                             (0, 192, 192, 192),
                                             expectedrows=data.n)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(cfg, sess)
        for i in range(min(data.n, cfg.MAX_STEPS)):
            print("%d/%d" % (i, data.n))
            blob = data.forward()
            if is_testing:
                blob['labels'] = blob['data'][..., 0]
            start = time.time()
            summary, results = net.test_image(sess, blob)
            end = time.time()
            duration += end - start
            # Drawing time
            # display_uresnet(blob, cfg, index=i, **results)
            if not is_testing:
                metrics.add(blob, results)
            mask = np.where(blob['data'][..., 0] > 0)
            preds = np.reshape(results['predictions'], (1, 192, 192, 192))
            print(np.count_nonzero(preds[mask] > 0))
            preds[mask] = 0
            preds_array.append(preds)
            print(preds.shape)

    preds_array.close()
    f_submission.close()

    duration /= cfg.MAX_STEPS
    print("Average duration of inference = %f ms" % duration)
    if not is_testing:
        metrics.plot()


if __name__ == '__main__':
    class MyCfg(object):
        IMAGE_SIZE = 192
        BASE_NET = 'uresnet'
        NET = 'base'
        ENABLE_CROP = False
        SLICE_SIZE = 64
        MAX_STEPS = 1
        CROP_ALGO = 'proba'
        DISPLAY_DIR = 'display/demo_codalab1'
        WEIGHTS_FILE_BASE = '/data/train_codalab1/model-145000.ckpt'
        DATA = '/data/codalab/train_5-6.csv'
        TEST_DATA = '/data/codalab/test_5-6.csv'
        DATA_TYPE = 'csv'
        GPU = '0'
        TOYDATA = False
        HDF5 = True
        DATA_3D = True
        SEED = 123
        NUM_CLASSES = 3
        LEARNING_RATE = 0.001
        BASE_NUM_OUTPUTS = 16
        WEIGHTS_FILE_PPN = None
        URESNET_WEIGHTING = False
        URESNET_ADD = False
        PPN2_INDEX = 3
        PPN1_INDEX = 1
        NUM_STRIDES = 3

    cfg = MyCfg()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    inference(cfg, is_testing=True)
