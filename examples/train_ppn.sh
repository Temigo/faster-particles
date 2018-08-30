# ---------
# TRAIN PPN
# ---------
#
# Run training of PPN with UResNet base network
# 2D, 512x512 images
# With NMS
# Load pre-trained weights for the base network and freeze the layers.
#
# Output:
# - folder display/train_ppn/train has event displays of PPN1 proposals and PPN2 predictions. Currently every 1000 steps.
# - logs for Tensorboard can be found in log/train_ppn
# - weights file are saved to output/train_ppn every 1000 steps
#
# Short version of the arguments can be found in config.py

python faster_particles/bin/ppn.py train \
	--display-dir display/train_ppn \
	--output-dir output/train_ppn \
	--log-dir log/train_ppn \
	--base-net uresnet \
	--net ppn \
	--weights-file-base /data/run_uresnet2d1/model-100000.ckpt \
	--freeze \
	--image-size 512 \
	--max-steps 10 \
	--data "/data/dlprod_ppn_v08_p01/train.root"
