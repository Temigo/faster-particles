# ------------
# DEMO URESNET
# ------------
#
# Run inference of UResNet
# 2D, 512x512 images
#
# Output:
# - folder demo/ has event displays of PPN1 proposals and PPN2 predictions.
#
# Short version of the arguments can be found in config.py

python faster_particles/bin/ppn.py demo \
	--display-dir display/demo_uresnet \
	--base-net uresnet \
	--net base \
	--weights-file-base /data/run_uresnet2d1/model-100000.ckpt \
	--image-size 512 \
	--max-steps 10 \
	--data "/data/dlprod_ppn_v08_p01/test_p01.root"
