# faster-particles
This package includes the following:
* Toydata generator
* LArCV data interface (2D and 3D)
* Pixel Proposal Network implementation using Tensorflow

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

### Install
Then install `faster-particles` with Pip:
```bash
pip install faster-particles
```

Alternatively, you can also clone the source:
```bash
git clone https://github.com/Temigo/faster-particles.git
cd faster-particles
```

## Usage

**The following assumes you installed with pip. If you cloned the source, make
sure you are in the root directory and replace `ppn` with `python faster_particles/bin/ppn.py`.**

To use toydata rather than LArCV data in the following sections, use option `--toydata`.
LArCV data files can be specified with `--data` option. They can use regex, e.g. `ppn_p[01]*.root`.

### Training
The program output is divided between:
* Output directory: with all the weights
* Log directory: to store all Tensorflow logs (and visualize them with Tensorboard)
* Display directory: stores regular snapshots taken during training of PPN1 and PPN2 proposals compared to ground truth.

To train PPN on 1000 steps use:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir -n ppn -m 1000 --data path/to/data
```

To train the base network (currently only VGG available) on track/shower classification task use:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir -n base -m 1000
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
