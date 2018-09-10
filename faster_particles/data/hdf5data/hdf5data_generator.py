from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tables
from faster_particles.display_utils import extract_voxels


class HDF5Generator(object):
    """
    Read HDF5 data files.
    Expecting at least a `data` column, possibly a `label` column.
    """

    def __init__(self, cfg, filelist="", is_testing=False):
        self.N = cfg.IMAGE_SIZE  # shape of canvas
        self.cfg = cfg
        self.dim = 3 if cfg.DATA_3D else 2
        self.is_testing = is_testing

        np.random.seed(cfg.SEED)
        self.file = tables.open_file(filelist, 'r')
        self.index = 0
        self.n = len(self.file.root.data)

    def forward(self):
        blob = {}
        blob['data'] = np.reshape(self.file.root.data[self.index], (1,) + (self.N,) * self.dim + (1,))
        if not self.is_testing:
            blob['labels'] = np.reshape(self.file.root.label[self.index], (1,) + (self.N,) * self.dim)
        blob['voxels'], blob['voxels_value'] = extract_voxels(blob['data'][0, ..., 0])
        blob['entries'] = [self.index]
        self.index = (self.index + 1) % self.n
        return blob
