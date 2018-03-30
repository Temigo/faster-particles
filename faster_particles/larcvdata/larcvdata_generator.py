from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imresize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

import ROOT
from larcv import larcv
larcv.ThreadProcessor
from larcv.dataloader2 import larcv_threadio
import tempfile

class LarcvGenerator(object):
    CLASSES = ('__background__', 'track_edge', 'shower_start', 'track_and_shower')

    def __init__(self, cfg, ioname="ThreadProcessor", filelist=""):
        self.N = cfg.IMAGE_SIZE # shape of canvas
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE

        np.random.seed(cfg.SEED)

        # FIXME random seed
        io_config = \
        """
%sIO: {
  Verbosity:    3
  EnableFilter: false
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   %s
  ProcessType:  ["BatchFillerImage2D","BatchFillerPPN","BatchFillerPPN"]
  ProcessName:  ["%s_data","%s_shower","%s_track"]
  NumThreads: 5
  NumBatchStorage: 5

  ProcessList: {
    %s_data: {
      Verbosity: 3
      ImageProducer: "data"
      Channels: [0]
    }
    %s_track: {
      Verbosity: 3
      ImageProducer: "data"
      ImageChannel: 0
      ParticleProducer: "ppn_mcst"
      BufferSize: 100
      ShapeType:  "track"
      PointType:  "xy"
    }
    %s_shower: {
      Verbosity: 3
      ImageProducer: "data"
      ImageChannel: 0
      ParticleProducer: "ppn_mcst"
      BufferSize: 100
      ShapeType:  "shower"
      PointType:  "xy"
    }
  }
}
        """ % ((ioname, filelist) + (ioname,)*6)
        # FIXME raises KeyError
        #io_config = io_config.format(ioname)
        self.ioname = ioname

        filler_config = tempfile.NamedTemporaryFile('w')
        filler_config.write(io_config)
        filler_config.flush()

        dataloader_cfg={}
        dataloader_cfg["filler_name"] = "%sIO" % ioname
        dataloader_cfg["verbosity"]   = 0,
        dataloader_cfg['filler_cfg']  = filler_config.name
        dataloader_cfg['make_copy']   = False # make explicit numpy array copy as we'll play w/ image data

        self.proc = larcv_threadio()
        self.proc.configure(dataloader_cfg)
        #self.proc.set_next_index(10345)
        self.proc.start_manager(self.batch_size)


    def __delete__(self):
        self.proc.stop_manager()
        self.proc.reset()

    def num_classes(self):
        return 4

    def forward(self):
        self.proc.next()
        batch_image  = self.proc.fetch_data ( '%s_data' % self.ioname   )
        batch_track  = self.proc.fetch_data ( '%s_track' % self.ioname  )
        batch_shower = self.proc.fetch_data ( '%s_shower' % self.ioname )

        gt_pixels = []
        output = []
        for index in np.arange(self.batch_size):
            image    = batch_image.data()  [index]
            t_points = batch_track.data()  [index]
            s_points = batch_shower.data() [index]
            # TODO set N from this
            #image_dim = batch_image.dim()
            #image = image.reshape(image_dim[1:3])
            #output.append(np.repeat(image.reshape([1, self.N, self.N, 1]), 3, axis=3)) # FIXME VGG needs RGB channels?

            for pt_index in np.arange(int(len(t_points)/2)):
                x = t_points[ 2*pt_index     ]
                y = t_points[ 2*pt_index + 1 ]
                if x < 0: break
                gt_pixels.append([y, x, 1])
            for pt_index in np.arange(int(len(s_points)/2)):
                x = s_points[ 2*pt_index     ]
                y = s_points[ 2*pt_index + 1 ]
                if x < 0: break
                gt_pixels.append([y, x, 2])
            if len(gt_pixels) > 0:
                output.append(image.reshape([1, self.N, self.N, 1]))

        if len(output) == 0: # No gt pixels in this batch - try next batch
            print("DUMP")
            return self.forward()

        # TODO For now we only consider batch size 1
        output = np.reshape(np.array(output), (1, self.N, self.N, 1))

        blob = {}
        blob['data'] = output.astype(np.float32)
        blob['im_info'] = [1, self.N, self.N, 3]
        blob['gt_pixels'] = np.array(gt_pixels)

        return blob

if __name__ == '__main__':
    class MyCfg:
        IMAGE_SIZE = 768
        SEED = 123
        BATCH_SIZE = 1

    t = LarcvGenerator(MyCfg(), ioname='test', filelist='["/data/drinkingkazu/dlprod_ppn_v05/ppn_p00_0000_0019.root"]')
    for i in range(10):
        blobdict = t.forward()
        print(blobdict['data'].shape)
        print("gt pixels shape ", blobdict['gt_pixels'].shape)
        print("gt pixels ", blobdict['gt_pixels'])
