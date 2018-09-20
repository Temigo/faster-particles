from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from faster_particles.ppn_utils import crop as crop_numpy
import time


class CropTest(tf.test.TestCase):
    def testCrop(self):
        crop_module = tf.load_op_library('./faster_particles/crop_op/crop_op.so')

        np.random.seed(123)
        tf.set_random_seed(123)
        N = 192
        MAX_STEPS = 10
        CROP_SIZE = 64
        im_proposals_np = (np.random.rand(N, N, N, 1) * 192.).astype(np.float32)
        crop_centers_np = np.random.randint(50, high=100, size=(100, 3))
        # crop_centers_np = np.array([[50, 50, 50]])
        # print(crop_centers_np.shape)
        # im_proposals = tf.constant(im_proposals_np.transpose(3, 2, 1, 0), dtype=tf.float32)
        im_proposals = tf.constant(im_proposals_np, dtype=tf.float32)
        crop_centers = tf.constant(crop_centers_np, dtype=tf.int32)
        crops = crop_module.crop(im_proposals, crop_centers, CROP_SIZE)
        # crops = tf.reverse(crops, 0)
        with self.test_session():
            duration = 0
            for i in range(MAX_STEPS):
                start = time.time()
                tf_result = crops.eval()
                end = time.time()
                duration += end - start
            print("TF duration = %f s" % (duration / MAX_STEPS))
            duration = 0
            for i in range(MAX_STEPS):
                start = time.time()
                np_result, _ = crop_numpy(crop_centers_np, CROP_SIZE, im_proposals_np[np.newaxis, ...], return_labels=False)
                end = time.time()
                duration += end - start
            print("NP duration = %f s" % (duration / MAX_STEPS))
            # print(tf_result.shape, np_result.shape)
            # print(tf_result, np_result)
            self.assertAllClose(tf_result, np_result)


if __name__ == "__main__":
  tf.test.main()
