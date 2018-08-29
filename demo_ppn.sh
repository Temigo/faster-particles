# --------
# DEMO PPN
# --------
#
# Run inference of PPN with UResNet base network
# 2D, 512x512 images
# With NMS
#
# Output:
# - folder demo/ has event displays of PPN1 proposals and PPN2 predictions.
# - folder metrics/ has CSV files and plots of PPN metrics.
#
# Short version of the arguments can be found in config.py
#
# Do not panick if you get a ValueError at the end. Some bug in metrics plot generation.

# 1. Run container first
# sh /u/ki/ldomine/sing /u/ki/ldomine/singularity_img/larcv2-singularity.simg

# 2. Set GPU
# export NV_GPU=0

# 3. Run PPN
python faster_particles/bin/ppn.py demo \
	--display-dir display/demo_ppn \
	--base-net uresnet \
	--net ppn \
	--weights-file-ppn /data/run_ppn_uresnet_corrected9/model-100000.ckpt \
	--image-size 512 \
	--max-steps 10000 \
	--data "/data/fuckgrid/fuckgrid/p*/larcv.root"