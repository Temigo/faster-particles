from algorithm import CroppingAlgorithm
import numpy as np

class Probabilistic(CroppingAlgorithm):
    def __init__(self, cfg):
        super(Probabilistic, self).__init__(cfg)

    def crop(self, coords):
        n = coords.shape[0]
        proba = np.ones((n)) / n
        i = 0
        max_patches = 100
        patches = [] # List of center coordinates of patches dxd
        voxel_num_boxes = np.zeros_like(proba)
        while np.count_nonzero(voxel_num_boxes < 2) > 0 and  i < max_patches:
            indices = np.random.choice(np.arange(n), p=proba)
            selection_inside = np.all(np.logical_and(coords >= coords[indices] - self.d/2, coords <= coords[indices] + self.d/2), axis=-1)
            indices_inside = np.argwhere(selection_inside)
            voxels_inside = coords[selection_inside]
            distances_to_center = np.sqrt(np.sum(np.power(voxels_inside - coords[indices], 2), axis=-1))
            new_proba = np.ones_like(proba)
            core_indices = distances_to_center <= self.a
            new_proba[indices_inside[core_indices]] = 0.01
            new_proba[indices_inside[np.logical_and(np.logical_not(core_indices), distances_to_center <= self.d)]] = 0.4

            voxel_num_boxes[indices_inside] += 1 # Update voxel_num_boxes: increment all voxels inside box
            patches.append(coords[indices])
            i += 1
            proba = proba * new_proba
            if np.sum(proba) == 0.0:
                break
            proba = proba / np.sum(proba)
        return np.array(patches), np.array([self.cfg.SLICE_SIZE] * len(patches))
