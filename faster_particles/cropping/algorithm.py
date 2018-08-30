import numpy as np
from faster_particles.ppn_utils import crop as crop_util


class CroppingAlgorithm(object):
    """
    Base class for any cropping algorithm, they should inherit from it
    and implement crop method (see below)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.d = cfg.SLICE_SIZE  # Patch or box/crop size
        self.a = cfg.SLICE_SIZE / 2  # Core size
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
        for i in range(len(patch_centers)):
            patch_center, patch_size = patch_centers[i], patch_sizes[i]
            blob = {}
            blob['data'], _ = crop_util(np.array([patch_center]),
                                        self.cfg.SLICE_SIZE,
                                        original_blob['data'])

            if 'labels' in original_blob:
                blob['labels'], _ = crop_util(np.array([patch_center]),
                                              self.cfg.SLICE_SIZE,
                                              original_blob['labels'][..., np.newaxis])

            # Flip patch_center coordinates
            # because gt_pixels coordinates are reversed
            # FIXME here or before blob['data'] ??
            patch_center = np.flipud(patch_center)
            # Select gt pixels
            if 'gt_pixels' in original_blob:
                indices = np.where(np.all(np.logical_and(
                    original_blob['gt_pixels'][:, :-1] >= patch_center - patch_size/2.0,
                    original_blob['gt_pixels'][:, :-1] < patch_center + patch_size/2.0), axis=1))
                blob['gt_pixels'] = original_blob['gt_pixels'][indices]
                blob['gt_pixels'][:, :-1] = blob['gt_pixels'][:, :-1] - (patch_center - patch_size / 2.0)

            # Select voxels
            # Flip patch_center coordinates back to normal
            patch_center = np.flipud(patch_center)
            if 'voxels' in original_blob:
                voxels = original_blob['voxels']
                blob['voxels'] = voxels[np.all(np.logical_and(
                    voxels >= patch_center - patch_size / 2.0,
                    voxels < patch_center + patch_size / 2.0), axis=1)]
                blob['voxels'] = blob['voxels'] - (patch_center - patch_size / 2.0)
                blob['entries'] = original_blob['entries']

            # Crops for small UResNet
            if self.cfg.NET == 'small_uresnet':
                blob['crops'], blob['crops_labels'] = crop_util(
                    blob['gt_pixels'][:, :-1],
                    self.cfg.CROP_SIZE, blob['data'])

            # FIXME FIXME FIXME
            # Make sure there is at least one ground truth pixel in the patch (for training)
            if self.cfg.NET not in ['ppn', 'ppn_ext', 'full'] or len(blob['gt_pixels']) > 0:
                batch_blobs.append(blob)

        return batch_blobs
