from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd


class CSVGenerator(object):
    """
    Import data from a CSV file formatted as in this example:
    event,label,val,x,y,z
    0,0.0,0.009038619,167,51,140
    0,2.0,0.58304226,140,70,189
    0,0.0,0.009021033,140,70,190
    0,1.0,0.022947583,140,71,177
    0,1.0,1.4886905,140,71,178
    0,1.0,0.5760929,140,71,179
    0,1.0,0.03625052,140,71,180
    0,2.0,0.017397273,140,71,184
    0,2.0,0.586628,140,71,185
    /!\ Supports only 3D data.
    """

    def __init__(self, cfg, filelist=""):
        self.N = cfg.IMAGE_SIZE  # shape of canvas
        self.cfg = cfg
        self.dim = 3 if cfg.DATA_3D else 2

        np.random.seed(cfg.SEED)
        self.index = 0

        df = pd.read_csv(filelist, delimiter=',')
        self.data = df.groupby('event')
        self.n = len(self.data)

    def forward(self):
        group = self.data.get_group(self.index)
        is_testing = 'label' not in group
        blob = {}
        blob['data'] = np.zeros((1,) + (self.N,) * self.dim + (1,),
                                dtype=np.float32)
        if not is_testing:
            blob['labels'] = np.zeros((1,) + (self.N,) * self.dim,
                                      dtype=np.int32)

        blob['voxels'] = np.stack([group.x.values, group.y.values, group.z.values], axis=-1)
        blob['voxels_value'] = group.val.values
        blob['data'][0, group.x.values, group.y.values, group.z.values, 0] = blob['voxels_value']
        if not is_testing:
            blob['labels'][0, group.x.values, group.y.values, group.z.values] = group.label.values
        blob['entries'] = [self.index]
        self.index = (self.index + 1) % self.n
        return blob
