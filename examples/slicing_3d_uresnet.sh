# -----------------------------
# TRAIN URESNET 3D WITH SLICING
# -----------------------------
#
# Run training of PPN with UResNet base network
# 3D, 512^3 images sliced into 128^3
# With NMS
# Load pre-trained weights for the base network and freeze the layers.
#
# Output:
# - folder display/train_uresnet3d/train has event displays of original/labels/predictions. Currently every 1000 steps.
# - logs for Tensorboard can be found in log/train_uresnet3d
# - weights file are saved to /data/train_uresnet3d every 1000 steps
#
# Short version of the arguments can be found in config.py

python faster_particles/bin/ppn.py train \
  --display-dir display/train_uresnet3d1 \
  -o /data/train_uresnet3d1 \
  -l log/train_uresnet3d1 \
  --base-net uresnet \
  --net base \
  --uresnet-weighting \
  --weights-file-base "/data/train_slicing9/model-40000.ckpt" \
  --data "/data/dlprod_ppn_v08_p02/train_p02.root" \
  --test-data "/data/dlprod_ppn_v08_p02/test_p02.root" \
  -N 192 \
  -m 1 \
  --enable-crop \
  -ss 64 \
  -3d \
  --gpu '5'
