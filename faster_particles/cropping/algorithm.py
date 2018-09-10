import numpy as np
from faster_particles.ppn_utils import crop as crop_util


class CroppingAlgorithm(object):
    """
    Base class for any cropping algorithm, they should inherit from it
    and implement crop method (see below)
    """

    def __init__(self, cfg, debug=False):
        self.cfg = cfg
        self.d = cfg.SLICE_SIZE  # Patch or box/crop size
        self.a = cfg.SLICE_SIZE / 2  # Core size
        self.N = cfg.IMAGE_SIZE
        self._debug = debug

    def crop(self, coords):
        """
        coords is expected to be dimensions (None, 3) = list of non-zero voxels
        Returns a list of patches centers and sizes (of cubes centered at the
        patch centers)
        """
        pass

    def process(self, original_blob):
        # FIXME cfg.SLICE_SIZE vs patch_size
        patch_centers, patch_sizes = self.crop(original_blob['voxels'])
        if self._debug:
            print("Cropping %d patches..." % len(patch_centers))
            print("Overlap: ",
                  self.compute_overlap(original_blob['voxels'],
                                       patch_centers,
                                       sizes=patch_sizes[:, np.newaxis]))
        batch_blobs = []
        for i in range(len(patch_centers)):
            patch_center, patch_size = patch_centers[i], patch_sizes[i]
            blob = {}

            # Flip patch_center coordinates
            # because gt_pixels coordinates are reversed
            # FIXME here or before blob['data'] ??
            patch_center = np.flipud(patch_center)

            blob['data'], _ = crop_util(np.array([patch_center]),
                                        self.cfg.SLICE_SIZE,
                                        original_blob['data'])
            patch_center = patch_center.astype(int)
            # print(patch_center, original_blob['data'][0, patch_center[0], patch_center[1], patch_center[2], 0], np.count_nonzero(blob['data']))
            # assert np.count_nonzero(blob['data']) > 0
            if 'labels' in original_blob:
                blob['labels'], _ = crop_util(np.array([patch_center]),
                                              self.cfg.SLICE_SIZE,
                                              original_blob['labels'][..., np.newaxis])
                blob['labels'] = blob['labels'][..., 0]
            # print(np.nonzero(blob['data']))
            # print(np.nonzero(blob['labels']))
            # assert np.array_equal(np.nonzero(blob['data']), np.nonzero(blob['labels']))
            if 'weight' in original_blob:
                blob['weight'], _ = crop_util(np.array([patch_center]),
                                              self.cfg.SLICE_SIZE,
                                              original_blob['weight'][..., np.newaxis])
                blob['weight'] = blob['weight'][..., 0]


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

        return batch_blobs, patch_centers, patch_sizes

    def compute_overlap(self, coords, patch_centers, sizes=None):
        """
        Compute overlap dict: dict[x] gives the number of voxels which belong
        to x patches.
        """
        if sizes is None:
            sizes = self.d/2.0
        overlap = []
        for voxel in coords:
            overlap.append(np.sum(np.all(np.logical_and(
                patch_centers-sizes <= voxel,
                patch_centers + sizes >= voxel
                ), axis=1)))
        return dict(zip(*np.unique(overlap, return_counts=True)))
