# faster-particles: Pixel Proposal Network (PPN) for particles images and related tools

## Introduction
This package includes the following:
* Toydata generator
* LArCV data interface (2D and 3D)
* Base network: VGG(ish) and UResNet
* Pixel Proposal Network implementation

## Contents
1. [Installation](#Installation)
  1.1. [Dependencies](#Dependencies)
  1.2. [Install](#Install)
2. [Usage](#Usage)
  2.1. [Dataset](#Dataset)

## License
This code is released under the MIT License (refer to the LICENSE file for more details).


## Installation
### Dependencies
You must install [larcv2](https://github.com/DeepLearnPhysics/larcv2) and its
own dependencies (ROOT, OpenCV, Numpy) in order to use LArCV data interface.
To install `larcv2`:
```bash
git clone https://github.com/DeepLearnPhysics/larcv2.git
cd larcv2
source configure.sh
make
```
You will also need [Tensorflow](http://tensorflow.org/).

### Install
The easiest way is to use Pip, although you will not get the latest changes:
```bash
pip install faster-particles
```

Alternatively, you can also clone the source if you want the latest updates or
participate to its development:
```bash
git clone https://github.com/Temigo/faster-particles.git
cd faster-particles
```

## Usage

**The following assumes you installed with pip. If you cloned the source, make
sure you are in the root directory and replace `ppn` with `python faster_particles/bin/ppn.py`.**

### Dataset
**Toydata**
To use toydata rather than LArCV data in the following sections, use the option `--toydata`.
*This is an old option which has not been tested for a while and which should be deprecated soon.*

**Liquid Argon data files**
LArCV data files should be specified with `--data` option which supports regex, e.g. `ppn_p[01]*.root`.
Some data files are publicly available at [DeepLearnPhysics](http://deeplearnphysics.org/DataChallenge/) data challenge page.

The generic usage is `ppn train/demo [directories options] [network architecture] [weights options] [network options] [other options]`. `train` is for training networks, `demo` is for running inference.

### Directories options
The program output is divided between:
* Output directory (option `-o`): with all the weights
* Log directory (option `-l`): to store all Tensorflow logs (and visualize them with Tensorboard)
* Display directory (option `-d`): stores regular snapshots taken during training of PPN1 and PPN2 proposals compared to ground truth.
The directories will be created if they do not exist yet. At training time all of them are required. At inference time only the display directory is required.

### Network architectures and weights options
#### Training
| Network trained | Command to run | Pretrained weights (optional) |
| --------------- | -------------------- | -------|
| Base network UResNet    | `--base-net uresnet --net base | `-wb uresnet.ckpt` |
| Base network VGG        | `--base-net vgg --net base` | `-wb vgg.ckpt` |
| PPN (w/ UResNet base)   | `--base-net uresnet --net ppn` | `-wp ppn.ckpt` |
| Small UResNet           | `--base-net uresnet --net small_uresnet` | `-ws small_uresnet.ckpt` |

### Inference
Use the command `ppn demo -d display/dir -m N_inferences` followed by:

| Network | Commandline options | Weights loading |
| --------|---------------------|-----------------|
| Base (UResNet)        | `--base-net uresnet --net base` | `--wb uresnet.ckpt` |
| PPN (w/ UResNet base) | `--base-net uresnet --net ppn`  | `--wp ppn.ckpt` |
| Small UResNet         | `--base-net uresnet --net small_uresnet` | `--ws model.ckpt` |
| PPN + UResNet         | `--base-net uresnet --net full` | `--wb uresnet.ckpt --wp ppn.ckpt` |
| PPN + Small UResNet   | `--base-net uresnet --net ppn_ext` | `--wp ppn.ckpt --ws small_uresnet.ckpt` |

### Most common options
|Option|Explanation|
|-----|----|
|`-m`| Number of steps / images to run on |
|`--freeze` | Freeze base network layers during training. |
|`-N` | Size of the image |
|`-3d`| 3D version |
|`-data`| Path to data files, can use wildcards and bash syntax. |

More options such as thresholds are available through `ppn train -h` and `ppn demo -h` respectively.

### Examples
To train PPN on 1000 steps use:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir --net ppn -m 1000 --data path/to/data
```

To train the base network (currently VGG and UResNet available) on track/shower classification task use:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir --net base --base-net vgg -m 1000
```

To train on 3D data, use the argument `-3d` and don't forget to specify the image size with `-N` argument (e.g. 192 for a compression factor of 4, see `larcvdata_generator.py` for more details).

To train PPN with UResNet base network pretrained weights, while freezing the base (pre-trained) layers,
 a more complete command line would be
```
ppn train -o output/dir/ -l log/dir/ -d display/dir --net ppn --base-net uresnet -wb /path/to/uresnet/weights --freeze -N 512 -m 100
```

To run inference with a minimal score of 0.5 for predicted points:
```bash
ppn demo weights_file.ckpt -d display/dir/ -ms 0.5
```
The display directory will contain snapshots of the results.


## Authors
K.Terao, J.W. Park, L.Domine
