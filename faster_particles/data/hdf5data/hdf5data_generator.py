from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tables


class HDF5Generator(object):

    def __init__(self, cfg, filelist=""):
        self.N = cfg.IMAGE_SIZE  # shape of canvas
        self.cfg = cfg
        self.dim = 3 if cfg.DATA_3D else 2

        np.random.seed(cfg.SEED)
        self.file = tables.open_file(filelist, 'r')
        self.index = 0
        self.n = len(self.file.root.data)

    def forward(self):
        blob = {}
        blob['data'] = np.reshape(self.file.root.data[self.index], (1,) + (self.N,) * self.dim + (1,))
        blob['labels'] = np.reshape(self.file.root.label[self.index], (1,) + (self.N,) * self.dim)
        self.index = (self.index + 1) % self.n
        return blob
