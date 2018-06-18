# faster-particles
This package includes the following:
* Toydata generator
* LArCV data interface (2D and 3D)
* Base network: VGG(ish) and UResNet
* Pixel Proposal Network implementation

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
You will also need Tensorflow.

### Install
The easiest way is to use Pip, although you will not get the latest changes:
```bash
pip install faster-particles
```

Alternatively, you can also clone the source if you want the latest updates or develop it:
```bash
git clone https://github.com/Temigo/faster-particles.git
cd faster-particles
```

## Usage

**The following assumes you installed with pip. If you cloned the source, make
sure you are in the root directory and replace `ppn` with `python faster_particles/bin/ppn.py`.**

### Dataset
To use toydata rather than LArCV data in the following sections, use option `--toydata`.
LArCV data files should be specified with `--data` option which supports regex, e.g. `ppn_p[01]*.root`.

### Training
The program output is divided between:
* Output directory: with all the weights
* Log directory: to store all Tensorflow logs (and visualize them with Tensorboard)
* Display directory: stores regular snapshots taken during training of PPN1 and PPN2 proposals compared to ground truth.
The directories will be created if they do not exist yet.

To train PPN on 1000 steps use:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir -n ppn -m 1000 --data path/to/data
```

To train the base network (currently VGG and UResNet available) on track/shower classification task use:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir --net base --base-net vgg -m 1000
```

To train on 3D data, use the argument `-3d` and don't forget to specify the image size with `-N` argument (e.g. 192 for a compression factor of 4, see `larcvdata_generator.py` for more details).

### Inference
To run inference with a minimal score of 0.5 for predicted points:
```bash
ppn demo weights_file.ckpt -d display/dir/ -ms 0.5
```
The display directory will contain snapshots of the results.

More options are available through `ppn train -h` and `ppn demo -h` respectively.

## Authors
K.Terao, J.W. Park, L.Domine
