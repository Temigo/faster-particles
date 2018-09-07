from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


class Metrics(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = os.path.join(cfg.DISPLAY_DIR, 'metrics')
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)

    def add(self, blob, results):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

    def make_plot(self, data, bins=None, xlabel="", ylabel="", filename=""):
        """
        If bins is None: discrete histogram
        """
        data = np.array(data)
        if bins is None:
            a = np.diff(np.unique(data))
            if len(a) > 0:
                d = a.min()
                left_of_first_bin = data.min() - float(d)/2
                right_of_last_bin = data.max() + float(d)/2
                bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
            else:
                bins = 100

        plt.hist(data, bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.dir, filename))
        plt.gcf().clear()
