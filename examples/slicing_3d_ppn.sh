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
  --display-dir display/train_ppn3d \
  -o /data/train_ppn3d \
  -l log/train_ppn3d \
  --base-net uresnet \
  --net ppn \
  --weights-file-base "/data/train_slicing9/model-40000.ckpt" \
  --freeze \
  --weights-file-ppn "/data/train_ppn3/model-112000.ckpt" \
  --data "/data/dlprod_ppn_v08_p01_filtered/train_p01.root" \
  --test-data "/data/dlprod_ppn_v08_p01_filtered/test_p01.root" \
  -N 512 \
  -m 1 \
  --enable-crop \
  -ss 128 \
  -3d
