from algorithm import CroppingAlgorithm
import numpy as np

class Octree(CroppingAlgorithm):
    def __init__(self, cfg):
        super(Octree, self).__init__(cfg)
        self.num_choice = 6
        self.final_size = 0.7 * cfg.CROP_SIZE

    def crop(self, coords):
        heap = [([self.N/2, self.N/2, self.N/2], self.N/2, coords)] # List
        patches = []
        sizes = []
        while len(heap) > 0:
            coordinates, size, voxels = heap.pop(0)
            x0, y0, z0 = coordinates
            new_size = size/2
            if size > self.final_size:
                new_cubes = [
                    ([x0 + new_size, y0 + new_size, z0 - new_size], new_size),
                    ([x0 - new_size, y0 + new_size, z0 - new_size], new_size),
                    ([x0 + new_size, y0 - new_size, z0 - new_size], new_size),
                    ([x0 - new_size, y0 - new_size, z0 - new_size], new_size),
                    ([x0 + new_size, y0 + new_size, z0 + new_size], new_size),
                    ([x0 - new_size, y0 + new_size, z0 + new_size], new_size),
                    ([x0 + new_size, y0 - new_size, z0 + new_size], new_size),
                    ([x0 - new_size, y0 - new_size, z0 + new_size], new_size)
                ]
                new_voxel_sets = []
                for cube, new_size in new_cubes:
                    if voxels.shape[0] > 0:
                        cube = np.array(cube)
                        indices = np.all(np.logical_and(voxels >= cube - new_size, voxels <= cube + new_size), axis=1)
                        new_voxel_sets.append(voxels[indices])
                        voxels = np.delete(voxels, indices, axis=0)

                for i, voxel_set in enumerate(new_voxel_sets):
                    if len(voxel_set) > 0:
                        heap.append(new_cubes[i] + (voxel_set,))
            else:
                #patches.append(coordinates)
                #sizes.append(size)
                new_cubes = [
                    [x0 + new_size, y0 + new_size, z0 - new_size],
                    [x0 - new_size, y0 + new_size, z0 - new_size],
                    [x0 + new_size, y0 - new_size, z0 - new_size],
                    [x0 - new_size, y0 - new_size, z0 - new_size],
                    [x0 + new_size, y0 + new_size, z0 + new_size],
                    [x0 - new_size, y0 + new_size, z0 + new_size],
                    [x0 + new_size, y0 - new_size, z0 + new_size],
                    [x0 - new_size, y0 - new_size, z0 + new_size],
                ]
                our_choice = np.unique(np.random.choice(np.arange(8), size=self.num_choice))
                new_cubes = np.take(new_cubes, our_choice, axis=0)
                patches.extend(new_cubes)
                #sizes.extend([size] * our_choice.shape[0])
                sizes.extend([d/2] * our_choice.shape[0])
        return np.array(patches), np.array(sizes)[:, None]
