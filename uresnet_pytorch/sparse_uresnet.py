from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import sparseconvnet as scn

# Accelerate *if all input sizes are same*
torch.backends.cudnn.benchmark = True


class UResNet(nn.Module):
    def __init__(self, is_3d, num_strides=3, base_num_outputs=16, num_classes=3, spatialSize=192):
        nn.Module.__init__(self)
        dimension = 3 if is_3d else 2
        reps = 2  # Conv block repetition factor
        kernel_size = 2  # Use input_spatial_size method for other values?
        m = base_num_outputs  # Unet number of features
        nPlanes = [i*m for i in range(1, num_strides+1)]  # UNet number of features per level
        # nPlanes = [(2**i) * m for i in range(1, num_strides+1)]  # UNet number of features per level
        nInputFeatures = 1
        self.sparseModel = scn.Sequential().add(
           scn.InputLayer(dimension, spatialSize, mode=3)).add(
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add(
           scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))

        self.linear = nn.Linear(m, num_classes)

    def forward(self, coords, features):
        # coords = (x > 0).nonzero()
        # coords_t = coords.transpose(0, 1)
        # features = x[coords_t[0], coords_t[1]]
        print(coords.size(), features.size())
        x = self.sparseModel((coords, features))
        print(x)
        x = self.linear(x)
        return x
