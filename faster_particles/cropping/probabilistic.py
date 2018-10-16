from algorithm import CroppingAlgorithm
import numpy as np


class Probabilistic(CroppingAlgorithm):
    """
    Probabilistic greedy cropping algorithm.
    ========================================
    Start with a flat probability distribution over all voxels.

    1. Select randomly a voxel according to the probability distribution.
    2. Mark all voxels inside the box centered at that voxel
    (mark differently the core and the sides)
    3. Update the probability distribution
    (becomes zero for the core for example)
    4. Start from 1. again until all voxels are marked properly
    (i.e. any voxel is either in a core or overlapped at least xx times)

    Guarantees a minimum coverage.
    """

    def __init__(self, cfg):
        super(Probabilistic, self).__init__(cfg)
        self.max_patches = cfg.MAX_PATCHES  # FIXME best value?
        self.min_overlap = cfg.MIN_OVERLAP

    def crop(self, coords):
        n = coords.shape[0]
        proba = np.ones((n)) / n
        i = 0
        patches = []  # List of center coordinates of patches dxd
        voxel_num_boxes = np.zeros_like(proba)
        voxel_num_cores = np.zeros_like(proba)
        while np.count_nonzero(voxel_num_cores < self.min_overlap) > 0 and i < self.max_patches:
            indices = np.random.choice(np.arange(n), p=proba)
            selection_inside = np.all(np.logical_and(
                coords >= coords[indices] - self.d/2,
                coords <= coords[indices] + self.d/2
                ), axis=-1)
            indices_inside = np.argwhere(selection_inside)
            voxels_inside = coords[selection_inside]
            distances_to_center = np.sqrt(np.sum(
                np.power(voxels_inside - coords[indices], 2),
                axis=-1))

            new_proba = np.ones_like(proba)
            core_indices = distances_to_center <= self.a
            new_proba[indices_inside[core_indices]] = 0.01
            new_proba[indices_inside[np.logical_and(
                np.logical_not(core_indices),
                distances_to_center <= self.d
                )]] = 0.4

            # Update voxel_num_boxes: increment all voxels inside box
            voxel_num_boxes[indices_inside] += 1
            voxel_num_cores[indices_inside[core_indices]] += 1
            patches.append(coords[indices])
            i += 1
            proba = proba * new_proba
            if np.sum(proba) == 0.0:
                break
            proba = proba / np.sum(proba)

        if i == self.max_patches:
            print("WARNING -- Reached the max number of patches in cropping algo.")

        return np.array(patches), np.array([self.cfg.SLICE_SIZE] * len(patches))
