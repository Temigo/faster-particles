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
        patch_centers, patch_sizes = self.crop(original_blob['voxels'])
        batch_blobs = []
        dim = 3 if self.cfg.DATA_3D else 2
        for i in range(len(patch_centers)):
            patch_center, patch_size = patch_centers[i], patch_sizes[i]
            # Flip patch_center coordinates because gt_pixels coordinates are reversed
            patch_center = np.flipud(patch_center)
            blob = {}
            blob['data'], _ = crop_util(np.array([patch_center]), self.cfg.SLICE_SIZE, original_blob['data'])
            indices = np.where(np.all(np.logical_and(original_blob['gt_pixels'][:, :-1] >= patch_center - patch_size/2.0, original_blob['gt_pixels'][:, :-1] <= patch_center + patch_size/2.0), axis=1))
            blob['gt_pixels'] = original_blob['gt_pixels'][indices]
            blob['gt_pixels'][:, :-1] = blob['gt_pixels'][:, :-1]  - (patch_center - patch_size / 2.0) - 1

            # Select voxels
            patch_center = np.flipud(patch_center) # Flip patch_center coordinates back to normal
            voxels = original_blob['voxels']
            blob['voxels'] = voxels[np.all(np.logical_and(voxels >= patch_center - patch_size / 2.0, voxels <= patch_center + patch_size / 2.0), axis=1)]
            blob['voxels'] = blob['voxels'] - (patch_center - patch_size / 2.0) - 1
            blob['entries'] = original_blob['entries']
            # FIXME FIXME FIXME
            if len(blob['gt_pixels']) > 0: # Make sure there is at least one ground truth pixel in the patch (for training)
                batch_blobs.append(blob)

        return batch_blobs
