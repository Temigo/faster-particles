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
# PyROOT hijacks help option otherwise
ROOT.PyConfig.IgnoreCommandLineOptions = True
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
        self.dim = 2
        if cfg.DATA_3D:
            self.dim = 3

        np.random.seed(cfg.SEED)

        # FIXME random seed
        if cfg.DATA_3D:
            io_config = \
            """
%sIO: {
  Verbosity:    3
  EnableFilter: true
  RandomAccess: 2
  RandomSeed:   123
  InputFiles:   %s
  ProcessType:  ["Tensor3DCompressor","BatchFillerTensor3D","BatchFillerPPN","BatchFillerPPN"]
  ProcessName:  ["%s_compressor","%s_data","%s_shower","%s_track"]
  NumThreads: 5
  NumBatchStorage: 5

  ProcessList: {
    %s_compressor: {
      Verbosity: 3
      Tensor3DProducer: "data"
      OutputProducer: "data"
      CompressionFactor: 4
      PoolType: 1
    }
    %s_data: {
      Verbosity: 3
      Tensor3DProducer: "data"
    }
    %s_track: {
      Verbosity: 3
      Tensor3DProducer: "data"
      ParticleProducer: "ppn_mcst"
      BufferSize: 20
      ShapeType:  "track"
      PointType:  "3d"
    }
    %s_shower: {
      Verbosity: 3
      Tensor3DProducer: "data"
      ParticleProducer: "ppn_mcst"
      BufferSize: 20
      ShapeType:  "shower"
      PointType:  "3d"
    }
  }
}
            """ % ((ioname, filelist) + (ioname,)*8)
        else:
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

        # Retrieve voxel sparse for track/shower only
        from ROOT import TChain
        self.ch = TChain("sparse3d_data_tree")
        self.ch.AddFile(cfg.DATA)

    def __delete__(self):
        self.proc.stop_manager()
        self.proc.reset()

    def num_classes(self):
        return 4

    def forward(self):
        self.proc.next(store_entries=True, store_event_ids=True)
        batch_image  = self.proc.fetch_data ( '%s_data' % self.ioname   )
        batch_track  = self.proc.fetch_data ( '%s_track' % self.ioname  )
        batch_shower = self.proc.fetch_data ( '%s_shower' % self.ioname )
        #batch_entries = self.proc.fetch_entries()
        #batch_event_ids = self.proc.fetch_event_ids()

        gt_pixels = []
        output = []
        img_shape = (1,) + (self.N,) * self.dim + (1,)
        for index in np.arange(self.batch_size):
            image    = batch_image.data()  [index]
            t_points = batch_track.data()  [index]
            s_points = batch_shower.data() [index]

            voxels = []

            """
            entry    = batch_entries.data()[index]
            event_id = batch_event_ids.data()
            self.ch.GetEntry(entry)
            event_data = self.ch.sparse3d_data_branch
            vox_array = event_data.as_vector()
            for vox in vox_array:
                pos = event_data.meta().position(vox.id())
                #print(vox.id(), vox.value(), '...', pos.x,pos.y,pos.z)
                #x, y, z = (self.N/768.0 * pos.x, self.N/768.0 * pos.y, self.N/768.0 * pos.z) # FIXME hardcoded
                x, y, z = pos.x, pos.y, pos.z
                x, y, z = (int(x), int(y), int(z))
                #voxels[x][y][z] = True # FIXME add value information
                voxels.append([x, y, z])
                # FIXME check this is the correct entry()"""

            indices = np.nonzero(image)[0]
            for i in indices:
                x = i%self.N
                i = (i-x)/self.N
                y = i%self.N
                i = (i-y)/self.N
                z = i%self.N
                voxels.append([x,y,z])
            image = image.reshape(img_shape)

            # TODO set N from this
            #image_dim = batch_image.dim()
            #image = image.reshape(image_dim[1:3])
            #output.append(np.repeat(image.reshape([1, self.N, self.N, 1]), 3, axis=3)) # FIXME VGG needs RGB channels?
            #print(t_points, s_points)
            if self.cfg.DATA_3D:
                for pt_index in np.arange(int(len(t_points)/3)):
                    x = t_points[ 3*pt_index     ]
                    y = t_points[ 3*pt_index + 1 ]
                    z = t_points[ 3*pt_index + 2 ]
                    if x < 0: break
                    gt_pixels.append([z, y, x, 1])
                for pt_index in np.arange(int(len(s_points)/3)):
                    x = s_points[ 3*pt_index     ]
                    y = s_points[ 3*pt_index + 1 ]
                    z = s_points[ 3*pt_index + 2 ]
                    if x < 0: break
                    gt_pixels.append([z, y, x, 2])
            else:
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
                output.append(image)

        if len(output) == 0: # No gt pixels in this batch - try next batch
            print("DUMP")
            return self.forward()

        # TODO For now we only consider batch size 1
        output = np.reshape(np.array(output), img_shape)

        blob = {}
        blob['data'] = output.astype(np.float32)
        blob['im_info'] = list(img_shape)
        blob['gt_pixels'] = np.array(gt_pixels)
        blob['voxels'] = np.array(voxels)
        return blob

if __name__ == '__main__':
    class MyCfg:
        IMAGE_SIZE = 192
        SEED = 123
        BATCH_SIZE = 1
        DATA_3D = True
        DATA = "/data/drinkingkazu/dlprod_ppn_v05/ppn_p00_0000_0019.root"

    t = LarcvGenerator(MyCfg(), ioname='test', filelist='["/data/drinkingkazu/dlprod_ppn_v05/ppn_p00_0000_0019.root"]')
    for i in range(2):
        blobdict = t.forward()
        print(blobdict['data'].shape)
        print("gt pixels shape ", blobdict['gt_pixels'].shape)
        print("gt pixels ", blobdict['gt_pixels'])
        print("voxels ", blobdict['voxels'])
