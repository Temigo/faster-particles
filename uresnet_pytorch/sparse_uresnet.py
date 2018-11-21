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
           scn.SubmanifoldConvolution(dimension, nInputFeatures, m, 3, False)).add( # Kernel size 3, no bias
           UNet(dimension, reps, nPlanes, residual_blocks=True, downsample=[kernel_size, 2])).add(  # downsample = [filter size, filter stride]
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, num_classes)

    def forward(self, coords, features):
        x = self.sparseModel((coords, features))
        x = self.linear(x)
        return x


def UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2, 2]):
    """
    U-Net style network with VGG or ResNet-style blocks.
    For voxel level prediction:
    import sparseconvnet as scn
    import torch.nn
    class Model(nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            self.sparseModel = scn.Sequential().add(
               scn.SubmanifoldConvolution(3, nInputFeatures, 64, 3, False)).add(
               scn.UNet(3, 2, [64, 128, 192, 256], residual_blocks=True, downsample=[2, 2]))
            self.linear = nn.Linear(64, nClasses)
        def forward(self,x):
            x=self.sparseModel(x).features
            x=self.linear(x)
            return x
    """
    def block(m, a, b):
        if residual_blocks: #ResNet style blocks
            m.add(scn.ConcatTable()
                  .add(scn.Identity() if a == b else scn.NetworkInNetwork(a, b, False))
                  .add(scn.Sequential()
                    .add(scn.BatchNormReLU(a))
                    .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False))
                    .add(scn.BatchNormReLU(b))
                    .add(scn.SubmanifoldConvolution(dimension, b, b, 3, False)))
             ).add(scn.AddTable())
        else: #VGG style blocks
            m.add(scn.Sequential()
                 .add(scn.BatchNormReLU(a))
                 .add(scn.SubmanifoldConvolution(dimension, a, b, 3, False)))

    def U(nPlanes): #Recursive function
        m = scn.Sequential()
        if len(nPlanes) == 1:
            for _ in range(reps):
                block(m, nPlanes[0], nPlanes[0])
        else:
            m = scn.Sequential()
            for _ in range(reps):
                block(m, nPlanes[0], nPlanes[0])
            m.add(
                scn.ConcatTable().add(
                    scn.Identity()).add(
                    scn.Sequential().add(
                        scn.BatchNormReLU(nPlanes[0])).add(
                        scn.Convolution(dimension, nPlanes[0], nPlanes[1],
                            downsample[0], downsample[1], False)).add(
                        U(nPlanes[1:])).add(
                        scn.BatchNormReLU(nPlanes[1])).add(
                        scn.Deconvolution(dimension, nPlanes[1], nPlanes[0],
                            downsample[0], downsample[1], False))))
            m.add(scn.JoinTable())
            for i in range(reps):
                block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
        return m

    m = U(nPlanes)
    return m
