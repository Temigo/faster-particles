# -------------------------
# TRAIN PPN 3D WITH SLICING
# -------------------------
#
# Run training of PPN with UResNet base network
# 3D, 512^3 images sliced into 128^3
# With NMS
# Load pre-trained weights for the base network and freeze the layers.
#
# Output:
# - folder display/train_ppn3d/train has event displays of PPN1 proposals and PPN2 predictions. Currently every 1000 steps.
# - logs for Tensorboard can be found in log/train_ppn3d
# - weights file are saved to /data/train_ppn3d every 1000 steps
#
# Short version of the arguments can be found in config.py

python faster_particles/bin/ppn.py train \
  --display-dir display/train_codalab \
  -o /data/train_codalab \
  -l log/train_codalab \
  --base-net uresnet \
  --net base \
  --data "/data/codalab/train_1-2.hdf5" \
  -N 192 \
  -m 100000 \
  -3d \
  -hdf5 \
  --gpu '2,3'
