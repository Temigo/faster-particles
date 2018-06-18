from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import scipy
from faster_particles.display_utils import display_original_image, display_im_proposals

class PPNMetrics(object):
    def __init__(self, cfg, dim1=8, dim2=4):
        self.im_labels = []
        self.im_scores = []
        self.im_proposals = []
        self.ppn1_gt_points_per_roi = []
        self.ppn1_ratio_gt_points_roi = []
        self.ppn1_distances_to_closest_gt = []
        self.ppn1_distances_to_closest_pred = []
        self.ppn1_ambiguity = []
        self.ppn1_false_positives = []
        self.ppn1_false_negatives = []
        self.ppn1_outliers = []
        self.ppn2_distances_to_closest_gt_raw = []
        self.ppn2_distances_to_closest_pred_raw = []
        self.ppn2_distances_to_closest_gt = []
        self.ppn2_distances_to_closest_pred = []
        self.ppn2_ambiguity = []
        self.ppn2_false_positives = []
        self.ppn2_false_negatives = []
        self.ppn2_outliers = []

        self.cfg = cfg
        self.dir = os.path.join(cfg.DISPLAY_DIR, 'metrics')
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        self.dim1, self.dim2 = dim1, dim2
        self.threshold_ambiguity = 10
        self.threshold_false_positive = 15
        self.threshold_false_negative = 15
        self.threshold_outliers = 15

    def add(self, blob, results, im_proposals_filtered):
        self.im_labels.extend(results['im_labels'])
        self.im_scores.extend(results['im_scores'])
        self.im_proposals.extend(results['im_proposals'])
        self.ppn1_gt_points_per_roi.extend(self.gt_points_per_roi(blob['gt_pixels'], results['rois']))
        self.ppn1_ratio_gt_points_roi.append(len(blob['gt_pixels'] / float(len(results['rois']))))

        # self.distances[i][j] = distance between im_proposals_filtered[j] and blob['gt_pixels'][i]
        gt_pixels = blob['gt_pixels'][:, :-1]
        im_proposals = results['im_proposals']
        distances_ppn2_raw = scipy.spatial.distance.cdist(im_proposals, gt_pixels)
        self.ppn2_distances_to_closest_gt_raw.extend(np.amin(distances_ppn2_raw, axis=0))
        self.ppn2_distances_to_closest_pred_raw.extend(np.amin(distances_ppn2_raw, axis=1))
        distances_ppn2 = scipy.spatial.distance.cdist(im_proposals_filtered, gt_pixels)
        self.ppn2_distances_to_closest_gt.extend(np.amin(distances_ppn2, axis=1))
        self.ppn2_distances_to_closest_pred.extend(np.amin(distances_ppn2, axis=0))
        self.ppn2_ambiguity.extend(np.count_nonzero(distances_ppn2 < self.threshold_ambiguity, axis=1))
        self.ppn2_false_positives.append(np.count_nonzero(np.all(distances_ppn2 > self.threshold_false_positive, axis=1)) / im_proposals_filtered.shape[0])
        self.ppn2_false_negatives.append(np.count_nonzero(np.all(distances_ppn2 > self.threshold_false_negative, axis=0)) / gt_pixels.shape[0])
        self.ppn2_outliers.append(np.count_nonzero(np.all(distances_ppn2 > self.threshold_outliers, axis=1)))

        #if np.logical_and(distances_ppn2 > 5, distances_ppn2 < 6).any():
        #    print(im_proposals_filtered, distances_ppn2)
        if np.logical_and(np.amin(distances_ppn2, axis=1) > 5, np.amin(distances_ppn2, axis=1) < 10).any():
            print(im_proposals_filtered, np.amin(distances_ppn2, axis=1))
            # --- FIGURE 2 : PPN2 predictions ---
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111, aspect='equal', projection='3d')
            display_original_image(blob, self.cfg, ax2, vmin=0, vmax=400, cmap='jet')
            im_proposals = display_im_proposals(self.cfg, ax2, im_proposals, results['im_scores'], results['im_labels'])
            ax2.set_xlim(0, self.cfg.IMAGE_SIZE)
            ax2.set_ylim(0, self.cfg.IMAGE_SIZE)
            ax2.set_zlim(0, self.cfg.IMAGE_SIZE)
            plt.show()

        rois = results['rois']
        if self.cfg.DATA_3D:
            rois[:, [0, 1, 2]] = rois[:, [2, 1, 0]]
        else:
            rois[:, [0, 1]] = rois[:, [1, 0]]
        # Go back to original coordinates and get center of ROI
        rois = self.dim1 * self.dim2 * rois + self.dim1

        distances_ppn1 = scipy.spatial.distance.cdist(rois, gt_pixels)
        self.ppn1_distances_to_closest_gt.extend(np.amin(distances_ppn1, axis=0))
        self.ppn1_distances_to_closest_pred.extend(np.amin(distances_ppn1, axis=1))
        self.ppn1_ambiguity.extend(np.count_nonzero(distances_ppn1 < self.threshold_ambiguity, axis=1))
        # FIXME rounding ROI to inner circle of radius self.dim1
        self.ppn1_false_positives.append(np.count_nonzero(np.all(distances_ppn1 > self.dim1, axis=1)) / rois.shape[0])
        self.ppn1_false_negatives.append(np.count_nonzero(np.all(distances_ppn1 > self.dim1, axis=0)) / gt_pixels.shape[0])
        self.ppn1_outliers.append(np.count_nonzero(np.all(distances_ppn1 > self.threshold_outliers, axis=1)))

    def plot(self):
        # Save data
        np.savetxt(os.path.join(self.dir, "ppn1_distances_to_closest_gt.csv"), self.ppn1_distances_to_closest_gt, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn1_distances_to_closest_pred.csv"), self.ppn1_distances_to_closest_pred, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn1_false_positives.csv"), self.ppn1_false_positives, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn1_false_negatives.csv"), self.ppn1_false_negatives, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn2_distances_to_closest_gt.csv"), self.ppn2_distances_to_closest_gt, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn2_distances_to_closest_pred.csv"), self.ppn2_distances_to_closest_pred, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn2_false_positives.csv"), self.ppn2_false_positives, delimiter=",")
        np.savetxt(os.path.join(self.dir, "ppn2_false_negatives.csv"), self.ppn2_false_negatives, delimiter=",")
        #self.ppn2_scores()
        self.plot_distances_to_closest_gt()
        self.plot_distances_to_closest_pred()
        self.plot_ambiguity()
        self.plot_false_positives()
        self.plot_false_negatives()
        self.plot_outliers()
        self.plot_gt_points_per_roi()
        self.plot_ratio_gt_points_roi()
        #self.ppn2_distance_to_nearest_neighbour()
        self.ppn2_distances_to_closest_gt_raw = np.array(self.ppn2_distances_to_closest_gt_raw)
        print("Mean of PPN2 distances to closest gt = ", np.mean(self.ppn2_distances_to_closest_gt))
        print("Mean of PPN2 distances to closest gt (raw) = ", np.mean(self.ppn2_distances_to_closest_gt_raw[np.where(self.ppn2_distances_to_closest_gt_raw < 5.0)]))

    def make_plot(self, data, bins=None, xlabel="", ylabel="", filename=""):
        """
        If bins is None: discrete histogram
        """
        data = np.array(data)
        if bins is None:
            d = np.diff(np.unique(data)).min()
            left_of_first_bin = data.min() - float(d)/2
            right_of_last_bin = data.max() + float(d)/2
            bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

        plt.hist(data, bins)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.dir, filename))
        plt.gcf().clear()

    def gt_points_per_roi(self, gt_pixels, rois):
        gt_points_per_roi_list = []
        for roi in rois:
            if self.cfg.DATA_3D:
                x, y, z = roi[2], roi[1], roi[0]
                x, y, z = x*self.dim1*self.dim2, y*self.dim2*self.dim1, z*self.dim1*self.dim2
                coords = np.array([x, y, z])
            else:
                x, y = roi[1], roi[0]
                x, y = x*self.dim1*self.dim2, y*self.dim1*self.dim2
                coords = np.array([x, y])
            gt_points_per_roi_list.append(np.count_nonzero(np.all(np.absolute(gt_pixels[:, :-1]  - coords) < self.dim1, axis=1)))
        return gt_points_per_roi_list

    def plot_distances_to_closest_gt(self):
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            self.ppn1_distances_to_closest_gt,
            bins=50,
            xlabel="distance to nearest ground truth pixel",
            ylabel="#ROI",
            filename='ppn1_distance_to_closest_gt.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_gt,
            bins=50,
            xlabel="distance to nearest ground truth pixel",
            ylabel="#proposed pixels",
            filename='ppn2_distance_to_closest_gt.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_gt_raw,
            bins=50,
            xlabel="distance to nearest ground truth pixel",
            ylabel="#proposed pixels",
            filename='ppn2_distance_to_closest_gt_raw.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_gt,
            bins=np.linspace(0, 5, 100),
            xlabel="distance to nearest ground truth pixel",
            ylabel="#proposed pixels",
            filename='ppn2_distance_to_closest_gt_zoom.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_gt_raw,
            bins=np.linspace(0, 5, 100),
            xlabel="distance to nearest ground truth pixel",
            ylabel="#proposed pixels",
            filename='ppn2_distance_to_closest_gt_raw_zoom.png'
        )

    def plot_distances_to_closest_pred(self):
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            self.ppn1_distances_to_closest_pred,
            bins=50,
            xlabel="distance to nearest ROI",
            ylabel="#ground truth pixels",
            filename='ppn1_distance_to_closest_pred.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_pred,
            bins=50,
            xlabel="distance to nearest proposed pixel",
            ylabel="#ground truth pixels",
            filename='ppn2_distance_to_closest_pred.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_pred_raw,
            bins=50,
            xlabel="distance to nearest proposed pixel",
            ylabel="#ground truth pixels",
            filename='ppn2_distance_to_closest_pred_raw.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_pred,
            bins=np.linspace(0, 5, 100),
            xlabel="distance to nearest proposed pixel",
            ylabel="#ground truth pixels",
            filename='ppn2_distance_to_closest_pred_zoom.png'
        )
        self.make_plot(
            self.ppn2_distances_to_closest_pred_raw,
            bins=np.linspace(0, 5, 100),
            xlabel="distance to nearest proposed pixel",
            ylabel="#ground truth pixels",
            filename='ppn2_distance_to_closest_pred_raw_zoom.png'
        )

    def plot_ambiguity(self):
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            self.ppn1_ambiguity,
            xlabel="#ground truth pixels within radius %d" % self.threshold_ambiguity,
            ylabel="#ROI",
            filename='ppn1_ambiguity.png',
            bins=None
        )
        self.make_plot(
            self.ppn2_ambiguity,
            xlabel="#ground truth pixels within radius %d" % self.threshold_ambiguity,
            ylabel="#proposed pixels",
            filename='ppn2_ambiguity.png',
            bins=None
        )

    def plot_false_positives(self):
        bins = np.arange(0, 1.1, 0.1)
        self.make_plot(
            self.ppn1_false_positives,
            bins=bins,
            xlabel="#empty ROI / #total ROI",
            ylabel="#images",
            filename='ppn1_false_positives.png'
        )
        self.make_plot(
            self.ppn2_false_positives,
            bins=bins,
            xlabel="#proposed points at a distance > %f from any ground truth point / #total proposed points" % self.threshold_false_positive,
            ylabel="#images",
            filename='ppn2_false_positives.png'
        )

    def plot_false_negatives(self):
        bins = np.arange(0, 1.1, 0.1)
        self.make_plot(
            self.ppn1_false_negatives,
            bins=bins,
            xlabel="#ground truth pixels outside of any ROI / #total ground truth pixels",
            ylabel="#images",
            filename='ppn1_false_negatives.png'
        )
        self.make_plot(
            self.ppn2_false_negatives,
            bins=bins,
            xlabel="#ground truth pixels at a distance > %f from any proposed point / #total ground truth points" % self.threshold_false_negative,
            ylabel="#images",
            filename='ppn2_false_negatives.png'
        )

    def plot_outliers(self):
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            self.ppn1_outliers,
            bins=10,
            xlabel="#ROI at a distance > %f from any ground truth point" % self.threshold_outliers,
            ylabel="#images",
            filename='ppn1_outliers.png'
        )
        self.make_plot(
            self.ppn2_outliers,
            bins=10,
            xlabel="#proposed points at a distance > %f from any ground truth point" % self.threshold_outliers,
            ylabel="#images",
            filename='ppn2_outliers.png'
        )

    def plot_gt_points_per_roi(self):
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            self.ppn1_gt_points_per_roi,
            bins=None,
            xlabel="#ground truth pixels / ROI",
            ylabel="#ROI",
            filename='ppn1_gt_points_per_roi.png'
        )

    def plot_ratio_gt_points_roi(self):
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            self.ppn1_ratio_gt_points_roi,
            bins=100,
            xlabel="#ground truth pixels / # ROI",
            ylabel="#images",
            filename='ppn1_ratio_gt_points_roi.png'
        )

    def ppn2_distance_to_nearest_neighbour(self):
        """
        Distance from a proposed point to
        its nearest neighbour (proposed point)
        """
        # FIXME vectorize loop
        distances = []
        for point in self.im_proposals:
            distances.append(np.partition(np.sum(np.power(point - self.im_proposals, 2), axis=1), 2)[1])
        bins = np.linspace(0, 100, 100)
        self.make_plot(
            distances,
            bins,
            xlabel="distance to nearest neighbour",
            ylabel="#proposals",
            filename='distance_to_nearest_neighbour.png'
        )
        return distances

    def ppn2_scores(self):
        """
        Histogram of scores of proposed points
        comparing track vs shower
        """
        track_scores = self.im_scores[self.im_scores == 0]
        shower_scores = self.im_scores[self.im_scores != 0]
        bins = np.linspace(0, 1, 20)
        plt.hist(track_scores, bins, alpha=0.5, label='track')
        plt.hist(shower_scores, bins, alpha=0.5, label='shower')
        plt.yscale('log', nonposy='clip')
        plt.legend(loc='upper right')
        plt.xlabel("Score")
        plt.ylabel("#Proposals")
        plt.savefig(os.path.join(self.dir, 'scores.png'))
