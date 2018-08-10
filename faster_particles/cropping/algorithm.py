import numpy as np
#from abc import ABC, abstractmethod
from faster_particles.ppn_utils import crop as crop_util


class CroppingAlgorithm(object):
    """
    Base class for any cropping algorithm, they should inherit from it
    and implement appropriate methods (see below)
    """
    def __init__(self, cfg):
        #super(CroppingAlgorithm, self).__init__()
        self.cfg = cfg
        self.d = cfg.SLICE_SIZE # Patch or box/crop size
        self.a = cfg.SLICE_SIZE / 2 # Core size
        self.N = cfg.IMAGE_SIZE

    def crop(self, coords):
        """
        coords is expected to be dimensions (None, 3) = list of non-zero voxels
        Returns a list of patches centers and sizes
        """
        pass

    def process(self, original_blob):
        print(original_blob)
        patch_centers, patch_sizes = self.crop(original_blob['voxels'])
        print(patch_centers, patch_sizes)
        # TODO Need to crop all the fields of blob including gt pixels
        batch_blobs = []
        dim = 3 if self.cfg.DATA_3D else 2 # FIXME below for 3D
        for i in range(len(patch_centers)):
            patch_center, patch_size = patch_centers[i], patch_sizes[i]
            blob = {}
            print(patch_center, patch_size)
            print(patch_center[0]-patch_size/2)
            #blob['data'] = original_blob['data'][0, int(patch_center[0]-patch_size/2):int(patch_center[0]+patch_size/2+1), (patch_center[1]-patch_size/2):(patch_center[1]+patch_size/2+1), 0]
            #blob['data'] = np.reshape(blob['data'], (1,)+ (self.cfg.SLICE_SIZE,) * dim + (1,))
            blob['data'], _ = crop_util(np.array([patch_center]), self.cfg.SLICE_SIZE, original_blob['data'])
            #blob['label'] = original_blob['label'][0, (patch_center[0]-patch_size[0]/2):(patch_center[0]+patch_size[0]/2+1), (patch_center[1]-patch_size[1]/2):(patch_center[1]+patch_size[1]/2+1)]
            #blob['label'] = np.reshape(blob['label'], (1,)+ (self.cfg.SLICE_SIZE,) * dim)
            indices = np.where(np.all(np.logical_and(original_blob['gt_pixels'][:, :-1] > patch_center - patch_size, original_blob['gt_pixels'][:, :-1] < patch_center + patch_size), axis=1))
            blob['gt_pixels'] = original_blob['gt_pixels'][indices]
            batch_blobs.append(blob)

        return batch_blobs
