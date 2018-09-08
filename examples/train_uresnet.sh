# -------------
# TRAIN URESNET
# -------------
#
# Run training of UResNet
# 2D, 512x512 images
#
# Output:
# - folder display/train_uresnet/train has event displays of original/labels/predictions. Currently every 1000 steps.
# - logs for Tensorboard can be found in log/train_uresnet
# - weights file are saved to output/train_uresnet every 1000 steps
#
# Short version of the arguments can be found in config.py

python faster_particles/bin/ppn.py train \
	--display-dir display/train_uresnet \
	--output-dir output/train_uresnet \
	--log-dir log/train_uresnet \
	--base-net uresnet \
	--net base \
	--image-size 512 \
	--max-steps 10 \
	--data "/data/dlprod_ppn_v08_p01_filtered/train_p01.root" \
  --test-data "/data/dlprod_ppn_v08_p01_filtered/test_p01.root"	
