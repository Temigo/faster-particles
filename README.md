# faster-particles

* Generate toy dataset
* Pixel Proposal Network implementation using Tensorflow

## Installation
With Pip [to be released soon]:
```bash
pip install faster-particles
```

You can also clone the source:
```bash
git clone https://github.com/Temigo/faster-particles.git
cd faster-particles/bin
```

## Usage

The following assumes either you installed with pip or you are in `bin` folder.
To train PPN on 1000 steps:
```bash
ppn train -o output/dir/ -l log/dir/ -d display/dir -n ppn -m 1000
```

To run inference:
```bash
ppn demo weights_file.ckpt -d display/dir/
```

More options available through `ppn train -h` and `ppn demo -h` respectively.

## Authors
K.Terao, J.W. Park, L.Domine
