# Define some base networks for PPN
# Currently only VGG is implemented.
# To add more base networks inherit from BaseNet class and update basenets variable.
import tensorflow as tf


class BaseNet(object):
    """
    Skeleton class from which any base network should inherit.
    PPN will assume this class structure for the base network.
    """
    def __init__(self, cfg, N=0):
        self.N = cfg.IMAGE_SIZE if N == 0 else N
        self.num_classes = cfg.NUM_CLASSES
        self.learning_rate = cfg.LEARNING_RATE
        self.is_3d = cfg.DATA_3D

    def init_placeholders(self):
        """
        Define placeholders *for the base net only* here as class attributes.
        @return: List of tuples (attribute_name, tf_name)
        """
        raise NotImplementedError

    def restore_placeholder(self, names):
        """
        For the test network to restore the placeholders previously created
        by the train network.
        @param names List of tuples (attribute_name, tf_name)
        """
        for attr, name in names:
            setattr(self, attr, tf.get_default_graph().get_tensor_by_name(name + ':0'))

    def test_image(self, sess, blob):
        """
        Run inference on a single image.
        @param sess Tensorflow current session.
        @param blob Data dictionary.
        @return: summary, dictionary of results
        """
        raise NotImplementedError

    def train_step_with_summary(self, sess, blobs):
        """
        Run training step.
        @param sess Tensorflow current session.
        @param blob Data dictionary.
        @return: summary, dictionary of results
        """
        _, summary = sess.run([self.train_op, self.summary_op], feed_dict=self.feed_dict(blobs))
        return summary, {}

    def feed_dict(self, blob):
        """
        Returns feed dict from blob.
        """
        raise NotImplementedError

    def build_base_net(self, image_placeholder, is_training=True, reuse=False):
        """
        Called by PPN and by the base net during training.
        @return: F3 and F5 layers
        """
        raise NotImplementedError

    def create_architecture(self, is_training=True, reuse=False, scope="base_net"):
        """
        Called to train the base network *only*. Can add layers to the base network used by PPN.
        @param scope Scope name to be shared between base network and PPN.
        """
        raise NotImplementedError
