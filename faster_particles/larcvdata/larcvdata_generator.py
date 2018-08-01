from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imresize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

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
        self.train_uresnet = (cfg.NET == 'base' and cfg.BASE_NET == 'uresnet')
        if cfg.DATA_3D:
            if self.train_uresnet:
                replace = 4
                config_file = 'uresnet_3d.cfg'
            elif self.cfg.NET == 'full':
                replace = 8
                config_file = 'ppn_uresnet_3d.cfg'
            else:
                replace = 6
                config_file = 'ppn_3d.cfg'
        else:
            if self.train_uresnet:
                replace = 4
                config_file = 'uresnet_2d.cfg'
            elif self.cfg.NET == 'full':
                replace = 8
                config_file = 'ppn_uresnet_2d.cfg'
            else:
                replace = 6
                config_file = 'ppn_2d.cfg'
        io_config = open(os.path.join(os.path.dirname(__file__), config_file)).read() % ((ioname, cfg.SEED, filelist) + (ioname,)*replace)
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
        self.proc.set_next_index(cfg.NEXT_INDEX)
        self.proc.start_manager(self.batch_size)
        self.proc.next()

    def __del__(self):
        self.proc.stop_manager()
        self.proc.reset()

    def forward_uresnet(self):
        self.proc.next()
        batch_image  = self.proc.fetch_data ( '%s_data' % self.ioname   )
        batch_labels  = self.proc.fetch_data ( '%s_labels' % self.ioname  )
        output_image, output_labels = [], []
        img_shape = (1,) + (self.N,) * self.dim + (1,)
        labels_shape = (1,) + (self.N,) * self.dim
        for index in np.arange(self.batch_size):
            image    = batch_image.data()  [index]
            labels   = batch_labels.data() [index]
            image = image.reshape(img_shape)
            labels = labels.reshape(labels_shape)
            output_image.append(image)
            output_labels.append(labels)
        # TODO For now we only consider batch size 1
        output_image = np.reshape(np.array(output_image), img_shape)
        output_labels = np.reshape(np.array(output_labels), labels_shape)
        blob = {}
        blob['data'] = output_image.astype(np.float32)
        blob['labels'] = output_labels.astype(np.int32)
        return blob

    def extract_voxels(self, image):
        voxels, voxels_value = [], []
        indices = np.nonzero(image)[0]
        for i in indices:
            x = i%self.N
            i = (i-x)/self.N
            y = i%self.N
            i = (i-y)/self.N
            z = i%self.N
            voxels.append([x,y,z])
            #voxels_value.append(image[i])
        return voxels

    def extract_gt_pixels(self, t_points, s_points):
        gt_pixels = []
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
        return gt_pixels

    def forward_ppn(self):
        self.proc.next(store_entries=True, store_event_ids=True)
        entries = self.proc.fetch_entries()
        batch_image  = self.proc.fetch_data ( '%s_data' % self.ioname   )
        batch_track  = self.proc.fetch_data ( '%s_track' % self.ioname  )
        batch_shower = self.proc.fetch_data ( '%s_shower' % self.ioname )
        #batch_entries = self.proc.fetch_entries()
        #batch_event_ids = self.proc.fetch_event_ids()

        gt_pixels, output, final_entries = [], [], []
        img_shape = (1,) + (self.N,) * self.dim + (1,)
        for index in np.arange(self.batch_size):
            image    = batch_image.data()  [index]
            t_points = batch_track.data()  [index]
            s_points = batch_shower.data() [index]
            entry_id = entries.data()      [index]

            final_entries.append(entry_id)
            voxels = self.extract_voxels(image)

            image = image.reshape(img_shape)

            # TODO set N from this
            gt_pixels.extend(self.extract_gt_pixels(t_points, s_points))
            if len(gt_pixels) > 0:
                output.append(image)

        if len(output) == 0: # No gt pixels in this batch - try next batch
            print("DUMP")
            return self.forward_ppn()

        # TODO For now we only consider batch size 1
        output = np.reshape(np.array(output), img_shape)

        blob = {}
        blob['data'] = output.astype(np.float32)
        blob['im_info'] = list(img_shape)
        blob['gt_pixels'] = np.array(gt_pixels)
        blob['voxels'] = np.array(voxels)
        blob['entries'] = final_entries
        return blob

    def forward_small_uresnet(self):
        blob = self.forward_ppn()
        # Crop regions around gt points
        # TODO add random
        N = self.cfg.CROP_SIZE
        coords0 = np.floor(blob['gt_pixels'][:, :-1] - N/2.0).astype(int)
        coords1 = np.floor(blob['gt_pixels'][:, :-1] + N/2.0).astype(int)
        dim = blob['gt_pixels'].shape[-1] - 1
        smear = np.random.randint(-12, high=12, size=dim)
        coords0 = np.clip(coords0 + smear, 0, self.cfg.IMAGE_SIZE-1)
        coords1 = np.clip(coords1 + smear, 0, self.cfg.IMAGE_SIZE-1)
        crops = np.zeros((coords0.shape[0], N, N))
        crops_labels = np.zeros_like(crops)
        for j in range(len(coords0)):
            padding = []
            for d in range(dim):
                pad = np.maximum(N - (coords1[j, d] - coords0[j, d]), 0)
                if coords0[j, d] == 0.0:
                    padding.append((pad, 0))
                else:
                    padding.append((0, pad))

            crops[j] = np.pad(blob['data'][0, coords0[j, 0]:coords1[j, 0], coords0[j, 1]:coords1[j, 1], 0], padding, 'constant')
            indices = np.where(crops[j] > 0)
            crops_labels[j][indices] = 1
            #crops_labels[j][indices[np.where(np.logical_and(indices >= int(N/2 - 1), indices <= int(N/2 + 1)))]] = 2

            indices = np.where(crops[j, int(N/2-1-smear[0]):int(N/2+1-smear[0]), int(N/2-1-smear[1]):int(N/2+1-smear[1])] > 0)
            a = indices[0] + int(N/2 - 1-smear[0])
            b = indices[1] + int(N/2 - 1-smear[1])
            #for a in indices:
            #    a = a + int(N/2 - 1)
            crops_labels[j][a, b] = 2

        blob['crops'] = crops
        blob['crops_labels'] = crops_labels
        return blob

    def forward_ppn_uresnet(self):
        self.proc.next(store_entries=True, store_event_ids=True)
        entries = self.proc.fetch_entries()
        batch_image  = self.proc.fetch_data ( '%s_data' % self.ioname   )
        batch_labels  = self.proc.fetch_data ( '%s_labels' % self.ioname  )
        batch_track  = self.proc.fetch_data ( '%s_track' % self.ioname  )
        batch_shower = self.proc.fetch_data ( '%s_shower' % self.ioname )

        gt_pixels = []
        output_image, output_labels = [], []
        img_shape = (1,) + (self.N,) * self.dim + (1,)
        labels_shape = (1,) + (self.N,) * self.dim
        for index in np.arange(self.batch_size):
            image    = batch_image.data()  [index]
            labels   = batch_labels.data() [index]
            t_points = batch_track.data()  [index]
            s_points = batch_shower.data() [index]

            voxels = self.extract_voxels(image)

            image = image.reshape(img_shape)
            labels = labels.reshape(labels_shape)

            # TODO set N from this
            #image_dim = batch_image.dim()
            #image = image.reshape(image_dim[1:3])
            gt_pixels_current = self.extract_gt_pixels(t_points, s_points)
            gt_pixels.extend(gt_pixels_current)
            if len(gt_pixels_current) > 0:
                output_image.append(image)
                output_labels.append(labels)

        if len(output_image) == 0: # No gt pixels in this batch - try next batch
            print("DUMP")
            return self.forward()

        # TODO For now we only consider batch size 1
        output_image = np.reshape(np.array(output_image), img_shape)
        output_labels = np.reshape(np.array(output_labels), labels_shape)

        blob = {}
        blob['data'] = output_image.astype(np.float32)
        blob['labels'] = output_labels.astype(np.int32)
        blob['im_info'] = list(img_shape)
        blob['gt_pixels'] = np.array(gt_pixels)
        blob['voxels'] = np.array(voxels)
        blob['entries'] = entries
        return blob

    def forward(self):
        # Now using a file with combined uresnet and ppn information
        if self.train_uresnet:
            return self.forward_uresnet()
        elif self.cfg.NET == 'full':
            return self.forward_ppn_uresnet()
        elif self.cfg.NET == 'small_uresnet':
            return self.forward_small_uresnet()
        else:
            return self.forward_ppn()

if __name__ == '__main__':
    class MyCfg:
        IMAGE_SIZE = 512
        SEED = 123
        BATCH_SIZE = 1
        DATA_3D = False
        NET = "small_uresnet"
        BASE = "uresnet"
        NEXT_INDEX = 0
        CROP_SIZE = 24

    t = LarcvGenerator(MyCfg(), ioname='test', filelist='["/stage/drinkingkazu/fuckgrid/p00/larcv.root"]')
    for i in range(1):
        blobdict = t.forward()
        print(blobdict['data'].shape)
        print("gt pixels shape ", blobdict['gt_pixels'].shape)
