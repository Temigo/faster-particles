
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from faster_particles.ppn_utils import filter_points

def draw_voxel(x, y, z, size, ax, alpha=0.3, facecolors='pink', **kwargs):
    vertices = [
        [[x, y, z], [x+size, y, z], [x, y+size, z], [x+size, y+size, z]],
        [[x, y, z+size], [x+size, y, z+size], [x, y+size, z+size], [x+size, y+size, z+size]],
        [[x, y, z], [x, y+size, z], [x, y, z+size], [x, y+size, z+size]],
        [[x+size, y, z], [x+size, y+size, z], [x+size, y, z+size], [x+size, y+size, z+size]],
        [[x, y, z], [x+size, y, z], [x, y, z+size], [x+size, y, z+size]],
        [[x, y+size, z], [x+size, y+size, z], [x, y+size, z+size], [x+size, y+size, z+size]]
    ]
    poly = Poly3DCollection(
        vertices,
        **kwargs
    )
    # Bug in Matplotlib with transparency of Poly3DCollection
    # see https://github.com/matplotlib/matplotlib/issues/10237
    poly.set_alpha(alpha)
    poly.set_facecolor(facecolors)
    ax.add_collection3d(poly)

def display_original_image(blob, cfg, ax, vmin=0, vmax=400, cmap='jet'):
    # Display original image
    if cfg.DATA_3D:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        colors = lambda i,j,k : matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap).to_rgba(blob['data'][0, i, j, k, 0])
        for i in range(len(blob['voxels'])):
            voxel = blob['voxels'][i]
            if 'voxels_value' in blob:
                if blob['voxels_value'][i] == 1: # track
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='teal', alpha=1.0, linewidths=0.0, edgecolors='black')
                elif blob['voxels_value'][i] == 2: # shower
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='gold', alpha=1.0, linewidths=0.0, edgecolors='black')
                else:
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors='black', alpha=0.3, linewidths=0.0, edgecolors='black')
            else:
                voxel = voxel.astype(int)
                c = colors(voxel[2], voxel[1], voxel[0])
                draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax, facecolors=c, alpha=1.0, linewidths=0.0, edgecolors=c)
    else:
        ax.imshow(blob['data'][0,...,0], cmap=cmap, interpolation='none', origin='lower', vmin=vmin, vmax=vmax)

def set_image_limits(cfg, ax):
    ax.set_xlim(0, cfg.IMAGE_SIZE)
    ax.set_ylim(0, cfg.IMAGE_SIZE)
    if cfg.DATA_3D:
        ax.set_zlim(0, cfg.IMAGE_SIZE)

def extract_voxels(data):
    indices = np.where(data > 0)
    return np.stack(indices).T, data[indices]

def display_im_proposals(cfg, ax, im_proposals, im_scores, im_labels):
    if im_proposals is not None and im_scores is not None:
        print(im_proposals)
        print(im_scores)
        """if len(im_proposals) > 0:
            eps = 20.0 #9.0
            if cfg.DATA_3D:
                eps = 15.0 # FIXME
            im_proposals = filter_points(im_proposals, im_scores, eps)"""

        for i in range(len(im_proposals)):
            proposal = im_proposals[i]
            #plt.text(proposal[1], proposal[0], str(im_scores[i][im_labels[i]]))
            if cfg.DATA_3D:
                x, y, z = proposal[2], proposal[1], proposal[0]
                ax.scatter([x], [y], [z], c='red')
            else:
                x, y = proposal[1], proposal[0]
                plt.plot([x], [y], 'ro')
    return im_proposals

def display_rois(cfg, ax, rois, dim1, dim2):
    if rois is not None:
        for roi in rois:
            if cfg.DATA_3D:
                x, y, z = roi[2], roi[1], roi[0]
                x, y, z = x*dim1*dim2, y*dim2*dim1, z*dim1*dim2
                size = dim1
                draw_voxel(x, y, z, size, ax,
                    facecolors='pink',
                    linewidths=0.01,
                    edgecolors='black',
                    alpha=0.1)
            else:
                x, y = roi[1], roi[0]
                ax.add_patch(
                    patches.Rectangle(
                        (x*dim1*dim2, y*dim1*dim2), # bottom left of rectangle
                        dim1, # width
                        dim1, # height
                        #fill=False,
                        #hatch='\\',
                        facecolor='pink',
                        alpha = 0.3,
                        linewidth=1.0,
                        edgecolor='black',
                    )
                )

def display_gt_pixels(cfg, ax, gt_pixels):
    if cfg.DATA_3D:
        for gt_pixel in gt_pixels:
            x, y, z = gt_pixel[2], gt_pixel[1], gt_pixel[0]
            draw_voxel(x, y, z, 1, ax, facecolors='red', alpha=1.0, linewidths=0.3, edgecolors='red')
    else:
        for gt_pixel in gt_pixels:
            x, y = gt_pixel[1], gt_pixel[0]
            if gt_pixel[2] == 1:
                plt.plot([x], [y], 'ro')
            elif gt_pixel[2] == 2:
                plt.plot([x], [y], 'go')

def display_uresnet(blob, cfg, index=0, predictions=None, name='display', directory='', vmin=0, vmax=400):
    if directory == '':
        directory = cfg.DISPLAY_DIR
    else:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'
        blob['voxels'], blob['voxels_value'] = extract_voxels(blob['data'][0,...,0])

    if predictions is not None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', **kwargs)
        display_original_image(blob, cfg, ax, vmin=vmin, vmax=vmax)
        set_image_limits(cfg, ax)
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_original_%d.png' % index), bbox_inches='tight')
        plt.close(fig)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
        blob_label = {}
        if cfg.DATA_3D:
            blob_label['data'] = blob['labels'][0,...]
            blob_label['voxels'], blob_label['voxels_value'] = extract_voxels(blob['labels'][0,...])
        else:
            blob_label['data'] = blob['labels'][:, :, :, np.newaxis]
        display_original_image(blob_label, cfg, ax2, vmax=np.unique(blob_label['data']).shape[0]-1, cmap='tab10')
        set_image_limits(cfg, ax2)
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_labels_%d.png' % index), bbox_inches='tight')
        plt.close(fig2)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, aspect='equal', **kwargs)
        blob_pred = {}
        if cfg.DATA_3D:
            blob_pred['data'] = predictions[0,...]
            blob_pred['voxels'], blob_pred['voxels_value'] = extract_voxels(predictions[0,...])
        else:
            blob_pred['data'] = predictions[:, :, :, np.newaxis]
        display_original_image(blob_pred, cfg, ax3, vmax=3.1)
        set_image_limits(cfg, ax3)
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_predictions_%d.png' % index), bbox_inches='tight')
        plt.close(fig3)


def display(blob, cfg, im_proposals=None, rois=None, im_labels=None, im_scores=None,
            index=0, dim1=8, dim2=4, name='display', directory=''):
    #print("gt_pixels: ", blob['gt_pixels'])
    #print("rois : ", rois*dim1*dim2)
    #print("im_proposals: ", im_proposals)
    #print("im_scores: ", im_scores)
    #print(im_labels)
    if directory == '':
        directory = cfg.DISPLAY_DIR
    else:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'

    # --- FIGURE 1 : PPN1 ROI ---
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', **kwargs)
    display_original_image(blob, cfg, ax, vmin=0, vmax=1)
    display_gt_pixels(cfg, ax, blob['gt_pixels'])
    display_rois(cfg, ax, rois, dim1, dim2)
    set_image_limits(cfg, ax)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(directory, name + '_proposals_%d_%d.png' % (index, blob['entries'][0])), bbox_inches='tight')
    plt.close(fig)

    # --- FIGURE 2 : PPN2 predictions ---
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
    display_original_image(blob, cfg, ax2, vmin=0, vmax=1, cmap='jet')
    im_proposals = display_im_proposals(cfg, ax2, im_proposals, im_scores, im_labels)
    set_image_limits(cfg, ax2)
    #plt.show()
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(directory, name + '_predictions_%d_%d.png' % (index, blob['entries'][0])), bbox_inches='tight')
    plt.close(fig2)
    return im_proposals

def display_ppn_uresnet(blob, cfg, im_proposals=None, rois=None, im_scores=None,
    index=0, dim1=8, dim2=4, predictions=None, im_labels=None, name='display', directory=''):
    if directory == '':
        directory = cfg.DISPLAY_DIR
    else:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'
        #blob['voxels'], blob['voxels_value'] = extract_voxels(blob['data'][0,...,0])

    plt.axis('off')

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', **kwargs)
    display_original_image(blob, cfg, ax, vmin=0, vmax=400, cmap='jet')
    set_image_limits(cfg, ax)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(directory, name + '_original_%d.png' % index), bbox_inches='tight', dpi=1000)
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
    blob_label = {}
    if cfg.DATA_3D:
        blob_label['data'] = blob['labels'][0,...]
        #blob_label['voxels'] = blob['voxels']
        #blob_label['voxels_value'] = blob_label['data'][blob['voxels'].T]
        blob_label['voxels'], blob_label['voxels_value'] = extract_voxels(blob['labels'][0,...])
        blob_label['voxels'][:, [0, 1, 2]] = blob_label['voxels'][:, [2, 1, 0]]
    else:
        blob_label['data'] = blob['labels'][:, :, :, np.newaxis]
    display_original_image(blob_label, cfg, ax2, vmax=3.1, cmap='tab10')
    display_gt_pixels(cfg, ax2, blob['gt_pixels'])
    set_image_limits(cfg, ax2)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(directory, name + '_labels_%d.png' % index), bbox_inches='tight', dpi=1000)
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, aspect='equal', **kwargs)
    blob_pred = {}
    if cfg.DATA_3D:
        blob_pred['data'] = predictions[0,...]
        blob_pred['voxels'], blob_pred['voxels_value'] = extract_voxels(predictions[0,...])
        blob_pred['voxels'][:, [0, 1, 2]] = blob_pred['voxels'][:, [2, 1, 0]]
    else:
        blob_pred['data'] = predictions[:, :, :, np.newaxis]
    display_original_image(blob_pred, cfg, ax3, vmax=3.1)
    display_im_proposals(cfg, ax3, im_proposals, im_scores, im_labels)
    set_image_limits(cfg, ax3)
    # Use dpi=1000 for high resolution
    plt.savefig(os.path.join(directory, name + '_predictions_%d.png' % index), bbox_inches='tight', dpi=1000)
    plt.close(fig3)

    return im_proposals
