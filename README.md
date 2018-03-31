# faster-particles
This package includes the following:
* Toydata generator
* LArCV data interface
* Pixel Proposal Network implementation using Tensorflow

## Installation
You must install [larcv2](https://github.com/DeepLearnPhysics/larcv2) and its
own dependencies (ROOT, OpenCV, Numpy) in order to use LArCV data interface.
To install `larcv2`:
```bash
git clone https://github.com/DeepLearnPhysics/larcv2.git
cd larcv2
source configure.sh
make
```

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
To train PPN on 1000 steps:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir -n ppn -m 1000 --data path/to/data
```

To train the base network (currently only VGG available) on track/shower classification task:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir -n base -m 1000
```


### Inference
To run inference with a minimal score of 0.5 for predicted points:
```bash
ppn demo weights_file.ckpt -d display/dir/ -ms 0.5
```

More options are available through `ppn train -h` and `ppn demo -h` respectively.

## Authors
K.Terao, J.W. Park, L.Domine
