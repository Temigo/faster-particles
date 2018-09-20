import tensorflow as tf
import numpy as np
from faster_particles.ppn_utils import crop as crop_numpy


class CropTest(tf.test.TestCase):
    def testCrop(self):
        crop_module = tf.load_op_library('./faster_particles/crop_op/crop_op.so')

        np.random.seed(123)
        tf.set_random_seed(123)

        im_proposals_np = (np.random.rand(192, 192, 192, 1) * 192.).astype(np.float32)
        crop_centers_np = np.array([[100, 100, 100]])

        im_proposals = tf.constant(im_proposals_np, dtype=tf.float32)
        crop_centers = tf.constant(crop_centers_np, dtype=tf.int32)
        crops = crop_module.crop(im_proposals, crop_centers, 64)
        with self.test_session():
            print(crops.eval())
            self.assertAllEqual(crops.eval(), crop_numpy(im_proposals, im_scores, 64))


if __name__ == "__main__":
  tf.test.main()
