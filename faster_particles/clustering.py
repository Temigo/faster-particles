# *-* encoding: utf-8 *-*
# Clustering

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator
from faster_particles.demo_ppn import get_filelist
from faster_particles.display_utils import display_original_image, display_gt_pixels

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

N = 40

def normalize_patch(patch):
    m1, m2 = np.amax(patch), np.amin(patch)
    return (patch - m2) / (m1 - m2)

def refine_point(cfg, blob, coords, index=0):
    directory = 'clustering2'
    x, y = coords # FIXME assuming 2d for now
    x, y = int(np.floor(x)), int(np.floor(y))
    if not (x - N/2 < 0 or x + N/2 > cfg.IMAGE_SIZE-1 or y-N/2 < 0 or y+N/2 > cfg.IMAGE_SIZE-1):
        x0 = int(np.maximum(0, x - N/2))
        x1 = int(np.minimum(cfg.IMAGE_SIZE-1, x + N/2))
        y0 = int(np.maximum(0, y - N/2))
        y1 = int(np.minimum(cfg.IMAGE_SIZE-1, y + N/2))
        print(x, y, x0, x1, y0, y1)



        zone = blob['data'][0,x0:x1, y0:y1,0]
        #print(np.amax(zone[zone > 0.0]), np.amin(zone[zone > 0.0]))
        zone = normalize_patch(zone)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal')
        p = ax2.imshow(zone, cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=1)
        fig2.colorbar(p)
        display_gt_pixels(cfg, ax2, [[x-x0, y-y0, 1]])

        threshold = 0.0
        border_spots = np.concatenate([
            np.transpose(np.where(zone[:3, :] > threshold)),
            np.transpose(np.where(zone[-3:, :] > threshold)),
            np.transpose(np.where(zone[:, :3] > threshold)),
            np.transpose(np.where(zone[:, -3:] > threshold))
            ])
        print(border_spots)
        if border_spots.size:
            db = DBSCAN(eps=10.0, min_samples=3).fit_predict(border_spots)
            print(db)

            unique_labels = set(db)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1: # noise
                    col = [0, 0, 0, 1]
                cluster = border_spots[db == k]
                plt.scatter(cluster[:, 1], N-1 - cluster[:, 0], color=col)

        plt.savefig(os.path.join(directory, '%d_%d.png' %  (index, blob['entries'][0])), bbox_inches='tight')
        plt.close(fig2)

def generate_cluster_data(cfg):
    filelist = get_filelist(cfg.DATA)
    data = LarcvGenerator(cfg, ioname="inference", filelist=filelist)
    directory = 'clustering_test'
    index = 0
    for i in range(cfg.MAX_STEPS):
        print("%d/%d" % (i, cfg.MAX_STEPS))
        blob = data.forward()
        print(blob['entries'])
        print(blob['gt_pixels'])
        for gt_pixel in blob['gt_pixels']:
            index += 1
            x, y, c = gt_pixel
            #refine_point(cfg, blob, (x, y), index=index)
            x, y = int(np.floor(x)), int(np.floor(y))
            #print(x, y, blob['data'][0, x, y, 0])

            if not (x - N/2 < 0 or x + N/2 > cfg.IMAGE_SIZE-1 or y-N/2 < 0 or y+N/2 > cfg.IMAGE_SIZE-1):
                x0 = int(np.maximum(0, x - N/2))
                x1 = int(np.minimum(cfg.IMAGE_SIZE-1, x + N/2))
                y0 = int(np.maximum(0, y - N/2))
                y1 = int(np.minimum(cfg.IMAGE_SIZE-1, y + N/2))
                #print(x, y, x0, x1, y0, y1)

                fig = plt.figure()
                ax = fig.add_subplot(111, aspect='equal')
                ax.imshow(blob['data'][0,x0:x1, y0:y1,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=1)
                plt.savefig(os.path.join(directory, '%d_%d.png' % (index, blob['entries'][0])), bbox_inches='tight')
                plt.close(fig)

                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111, aspect='equal')
                ax3.imshow(blob['data'][0,...,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=1)
                display_gt_pixels(cfg, ax3, blob['gt_pixels'])
                plt.savefig(os.path.join(directory, '%d_%d_original.png' % (index, blob['entries'][0])), bbox_inches='tight')
                plt.close(fig3)

                fig4 = plt.figure()
                ax4 = fig4.add_subplot(111, aspect='equal')
                ax4.imshow(blob['data'][0,...,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=1)
                plt.savefig(os.path.join(directory, '%d_%d_original_no_gt.png' % (index, blob['entries'][0])), bbox_inches='tight')
                plt.close(fig4)

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111, aspect='equal')
                ax2.imshow(blob['data'][0,x0:x1, y0:y1,0], cmap='jet', interpolation='none', origin='lower', vmin=0, vmax=1)
                display_gt_pixels(cfg, ax2, [[x-x0, y-y0, c]])
                plt.savefig(os.path.join(directory, '%d_%d_gt.png' %  (index, blob['entries'][0])), bbox_inches='tight')
                plt.close(fig2)


    """net = PPN(cfg=cfg, base_net=basenets[cfg.BASE_NET])
    if cfg.WEIGHTS_FILE_PPN is None:
        raise Exception("No weights file for PPN!")

    net.init_placeholders()
    net.create_architecture(is_training=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(cfg, sess)

        for i in range(cfg.MAX_STEPS):
            print("%d/%d" % (i, cfg.MAX_STEPS))
            blob = data.forward()
            summary, results = net.test_image(sess, blob)

            im_proposals_filtered = display(
                blob,
                cfg,
                index=i,
                dim1=net.dim1,
                dim2=net.dim2,
                directory=os.path.join(cfg.DISPLAY_DIR, 'demo'),
                **results
            )     """



if  __name__ == '__main__':
    class MyCfg(object):
        #DATA = "/stage/drinkingkazu/dlprod_ppn_v06/blur_train.root"
        #DATA = "/home/drinkingkazu/larcv.root"
        DATA = "/home/drinkingkazu/cacca8.root"
        MAX_STEPS = 100
        IMAGE_SIZE = 512#384
        BATCH_SIZE = 1
        DATA_3D = False
        SEED = 123
        NET = 'ppn'
        BASE_NET = 'uresnet'
        NEXT_INDEX = 0

    cfg = MyCfg()
    generate_cluster_data(cfg)
