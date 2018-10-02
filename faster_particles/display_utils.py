from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


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
    """
    Display original image.

    If data is 2D or no voxel_values are provided, `data` is mandatory.
    IF `voxel_values` is provided it will be used for color.
    """
    if cfg.DATA_3D:
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        colorbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        if 'voxels_value' in blob:
            colors = lambda i: colorbar.to_rgba(blob['voxels_value'][i])
        else:
            colors = lambda i, j, k: colorbar.to_rgba(blob['data'][0, i, j, k, 0])
        for i, voxel in enumerate(blob['voxels']):
            if 'voxels_value' in blob:
                if blob['voxels_value'][i] == 1:  # track
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax,
                               facecolors='teal', alpha=1.0,
                               linewidths=0.0, edgecolors='black')
                elif blob['voxels_value'][i] == 2:  # shower
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax,
                               facecolors='gold', alpha=1.0,
                               linewidths=0.0, edgecolors='black')
                else:
                    c = colors(i)
                    draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax,
                               facecolors=c, alpha=1.0,
                               linewidths=0.0, edgecolors=c)
            else:
                voxel = voxel.astype(int)
                c = colors(voxel[2], voxel[1], voxel[0])
                draw_voxel(voxel[0], voxel[1], voxel[2], 1, ax,
                           facecolors=c, alpha=1.0,
                           linewidths=0.0, edgecolors=c)
        return colorbar
    else:
        return ax.imshow(blob['data'][0, ..., 0], cmap=cmap,
                         interpolation='none', origin='lower',
                         vmin=vmin, vmax=vmax)


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


def display_uresnet(blob, cfg, index=0, predictions=None, scores=None,
                    name='display', directory='', vmin=0, vmax=400,
                    softmax=None, compute_voxels=True):
    if directory == '':
        directory = cfg.DISPLAY_DIR
    else:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'
        if compute_voxels:
            blob['voxels'], blob['voxels_value'] = extract_voxels(blob['data'][0,...,0])

    display_blob(blob, cfg, directory=directory, name=name, index=index, **kwargs)

    display_labels(blob, cfg, directory=directory, name=name, index=index, **kwargs)

    if 'weight' in blob:
        print("-- Weights:")
        blob['weight'][blob['weight'] <= 1.0] = 0.0
        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111, aspect='equal', **kwargs)
        blob_weight = {}
        if cfg.DATA_3D:
            blob_weight['data'] = blob['weight'][0, ...]
            blob_weight['voxels'], blob_weight['voxels_value'] = extract_voxels(blob['weight'][0, ...])
        else:
            blob_weight['data'] = blob['weight'][:, :, :, np.newaxis]
        display_original_image(blob_weight, cfg, ax5, vmax=np.amax(blob['weight']))
        set_image_limits(cfg, ax5)
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_weight_%d.png' % index), bbox_inches='tight')
        plt.close(fig5)
        print("-- OK.")

    display_predictions(blob, cfg, predictions, directory=directory, name=name, index=index, **kwargs)

    # if scores is not None:
    #     fig4 = plt.figure()
    #     ax4 = fig4.add_subplot(111, aspect='equal', **kwargs)
    #     blob_pred = {}
    #     if cfg.DATA_3D:
    #         blob_pred['data'] = scores[0,...]
    #         blob_pred['voxels'], blob_pred['voxels_value'] = extract_voxels(scores[0,...])
    #     else:
    #         blob_pred['data'] = scores[:, :, :, np.newaxis]
    #     display_original_image(blob_pred, cfg, ax4, vmax=1.0)
    #     set_image_limits(cfg, ax4)
    #     # Use dpi=1000 for high resolution
    #     plt.savefig(os.path.join(directory, name + '_scores_%d.png' % index), bbox_inches='tight')
    #     plt.close(fig4)


def display(blob, cfg, im_proposals=None, rois=None, im_labels=None, im_scores=None,
            index=0, dim1=8, dim2=4, name='display', directory=''):
    print("gt_pixels: ", blob['gt_pixels'])
    print("im_proposals: ", im_proposals)
    print("im_scores: ", im_scores)

    if directory == '':
        directory = cfg.DISPLAY_DIR
    else:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'

    # --- FIGURE 1 : PPN1 ROI ---
    display_blob_rois(blob, cfg, rois, dim1, dim2, directory=directory, name=name, index=index, **kwargs)

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


def display_labels(blob, cfg, directory=None, name='display', index=0, **kwargs):
    if 'labels' in blob:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
        blob_label = {}
        if cfg.DATA_3D:
            blob_label['data'] = blob['labels'][0, ...]
            blob_label['voxels'], blob_label['voxels_value'] = extract_voxels(blob['labels'][0, ...])
            blob_label['voxels'][:, [0, 1, 2]] = blob_label['voxels'][:, [2, 1, 0]]
        else:
            blob_label['data'] = blob['labels'][:, :, :, np.newaxis]
        display_original_image(blob_label, cfg, ax2, vmax=np.unique(blob_label['data']).shape[0]-1, cmap='tab10')
        if 'gt_pixels' in blob:
            display_gt_pixels(cfg, ax2, blob['gt_pixels'])
        set_image_limits(cfg, ax2)
        if directory is None:
            plt.show()
        else:
            # Use dpi=1000 for high resolution
            plt.savefig(os.path.join(directory, name + '_labels_%d.png' % index), bbox_inches='tight')
        plt.close(fig2)


def display_predictions(blob, cfg, predictions, im_proposals=None, im_scores=None, im_labels=None, directory=None, name='display', index=0, **kwargs):
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
    if im_proposals is not None and im_scores is not None and im_labels is not None:
        display_im_proposals(cfg, ax3, im_proposals, im_scores, im_labels)
    set_image_limits(cfg, ax3)
    if directory is None:
        plt.show()
    else:
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_predictions_%d.png' % index), bbox_inches='tight')
    plt.close(fig3)


def display_blob(blob, cfg, directory=None, name='display', index=0, cmap='jet', **kwargs):
    fig = plt.figure()
    if cfg.DATA_3D and ('voxels' not in blob or 'voxels_value' not in blob):
        blob['voxels'], blob['voxels_value'] = extract_voxels(blob['data'][0, ..., 0])
        blob['voxels'][:, [0, 1, 2]] = blob['voxels'][:, [2, 1, 0]]
    ax = fig.add_subplot(111, aspect='equal', **kwargs)
    display_original_image(blob, cfg, ax, vmin=np.amin(blob['voxels_value']), vmax=np.amax(blob['voxels_value']), cmap=cmap)
    set_image_limits(cfg, ax)
    if directory is None:
        plt.show()
    else:
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_original_%d.png' % index), bbox_inches='tight')
    plt.close(fig)


def display_blob_rois(blob, cfg, rois, dim1, dim2, directory=None, name='display', index=0, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', **kwargs)
    display_original_image(blob, cfg, ax, vmin=0, vmax=1)
    display_gt_pixels(cfg, ax, blob['gt_pixels'])
    display_rois(cfg, ax, rois, dim1, dim2)
    set_image_limits(cfg, ax)
    if directory is None:
        plt.show()
    else:
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_proposals_%d_%d.png' % (index, blob['entries'][0])), bbox_inches='tight')
    plt.close(fig)


def display_ppn_uresnet(blob, cfg, im_proposals=None, rois=None, im_scores=None,
    index=0, dim1=8, dim2=4, predictions=None, im_labels=None, name='display',
    directory=None, softmax=None, scores=None):
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

    display_blob(blob, cfg, directory=directory, name=name, index=index, **kwargs)

    display_labels(blob, cfg, directory=directory, name=name, index=index, **kwargs)

    display_predictions(blob, cfg, predictions, im_proposals=im_proposals, im_scores=im_scores, im_labels=im_labels, directory=directory, name=name, index=index, **kwargs)

    return im_proposals


def draw_cube(ax, coord, size):
    x, y, z = coord[0], coord[1], coord[2]
    vertices = [
        [[x, y, z], [x, y+size, z], [x+size, y+size, z], [x+size, y, z]],
        [[x, y, z+size], [x, y+size, z+size], [x+size, y+size, z+size], [x+size, y, z+size]],
        [[x, y, z], [x, y+size, z], [x, y+size, z+size], [x, y, z+size]],
        [[x+size, y, z], [x+size, y+size, z], [x+size, y+size, z+size], [x+size, y, z+size]],
        [[x, y, z], [x+size, y, z], [x+size, y, z+size], [x, y, z+size]],
        [[x, y+size, z], [x+size, y+size, z], [x+size, y+size, z+size], [x, y+size, z+size]]
    ]
    poly = Poly3DCollection(vertices, linewidths=0.1, edgecolors='r')
    poly.set_alpha(0)
    poly.set_facecolors('cyan')
    ax.add_collection3d(poly)


def compute_voxel_overlap(coords, patch_centers, patch_sizes):
    """
    Returns overlap value for each voxel.
    """
    overlap = []
    for voxel in coords:
        overlap.append(np.sum(np.all(np.logical_and(
            patch_centers-patch_sizes/2.0 <= voxel,
            patch_centers + patch_sizes/2.0 >= voxel
            ), axis=1)))
    return overlap


def compute_voxel_core(coords, patch_centers, core_size):
    """
    Returns for each voxel whether it belongs to a core region.
    """
    overlap = []
    for voxel in coords:
        overlap.append(np.any(np.all(np.logical_and(
            patch_centers - core_size/2.0 <= voxel,
            patch_centers + core_size/2.0 >= voxel
            ), axis=1)))
    return overlap


def draw_slicing(blob, cfg, patch_centers, patch_sizes,
                 name='display', directory=None, index=0):

    if directory is not None:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'

    # 1. Plot original image with patches drawn as red cubes
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', **kwargs)
    display_original_image(blob, cfg, ax, vmin=0, vmax=1)
    for i, p in enumerate(patch_centers):
        if type(patch_sizes) == int or type(patch_sizes) == float:
            draw_cube(ax, p - patch_sizes/2.0, patch_sizes)
        else:
            draw_cube(ax, p - patch_sizes[i]/2.0, patch_sizes[i])
    set_image_limits(cfg, ax)
    if directory is None:
        print("Crops")
        plt.show()
    else:
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory,
                                 name + '_patches_%d_%d.png' % (index, blob['entries'][0])),
                    bbox_inches='tight')
    plt.close(fig)

    # 2. Plot overlap values for each voxel
    overlap = compute_voxel_overlap(blob['voxels'], patch_centers, patch_sizes[:, np.newaxis])
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal', **kwargs)
    blob_overlap = {}
    if cfg.DATA_3D:
        blob_overlap['voxels'], blob_overlap['voxels_value'] = blob['voxels'], overlap
    else:
        blob_overlap['data'] = blob['data']
    colorbar = display_original_image(blob_overlap, cfg, ax2,
                                      vmax=np.amax(overlap), cmap='rainbow')
    colorbar.set_array([])
    fig2.colorbar(colorbar, ax=ax2)
    set_image_limits(cfg, ax2)
    if directory is None:
        print("Number of crops to which a voxel belongs (overlap)")
        plt.show()
    else:
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_overlap_%d.png' % index),
                    bbox_inches='tight')
    plt.close(fig2)

    # 3. Whether each voxel belongs to at least 1 core region
    overlap = compute_voxel_core(blob['voxels'], patch_centers, cfg.CORE_SIZE)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, aspect='equal', **kwargs)
    blob_core = {}
    if cfg.DATA_3D:
        blob_core['voxels'], blob_core['voxels_value'] = blob['voxels'], overlap
    else:
        blob_core['data'] = blob['data']
    colorbar = display_original_image(blob_core, cfg, ax3,
                                      vmax=np.amax(overlap), cmap='tab10')
    # colorbar.set_array([])
    # fig2.colorbar(colorbar, ax=ax2)
    set_image_limits(cfg, ax3)
    if directory is None:
        pass
    else:
        # Use dpi=1000 for high resolution
        plt.savefig(os.path.join(directory, name + '_core_%d.png' % index),
                    bbox_inches='tight')
    plt.close(fig3)
