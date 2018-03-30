# *-* encoding: utf-8 *-*
# Generate toy dataset
# with labels = feature space points (track and shower start/end points)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.misc import imresize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from faster_particles.toydata.track_generator import generate_toy_tracks
from faster_particles.toydata.shower_generator import make_shower

class ToydataGenerator(object):
    CLASSES = ('__background__', 'track_edge', 'shower_start', 'track_and_shower')

    def __init__(self, cfg, classification=False):
        self.N = cfg.IMAGE_SIZE # shape of canvas

        # Track options
        self.max_tracks = cfg.MAX_TRACKS
        self.max_kinks = cfg.MAX_KINKS
        self.max_track_length = cfg.MAX_TRACK_LENGTH
        self.kinks = cfg.KINKS

        # Shower options
        self.max_showers = cfg.MAX_SHOWERS
        self.cfg = cfg
        self.gt_box_padding = 5
        self.batch_size = cfg.BATCH_SIZE
        self.classification = classification
        if classification:
            self.max_tracks = 1
        np.random.seed(cfg.SEED)

    def num_classes(self):
        return 4

    def make_showers(self):
        output_showers, shower_start_points, angle = np.zeros((self.N, self.N)), [], []
        for i in range(np.random.randint(self.max_showers)):
            scale = np.random.uniform(0.3, 1.0)
            output_showers_i, shower_start_points_i, angle_i = make_shower(self.cfg)
            shower_image = imresize(output_showers_i, scale)
            #print(shower_image.shape, scale)
            #print(shower_image)
            #plt.imshow(shower_image)
            #plt.savefig(self.cfg.DISPLAY_DIR + "/shower%d.png" % i)
            before_1 = np.random.randint(self.cfg.IMAGE_SIZE - shower_image.shape[0])
            after_1 = self.cfg.IMAGE_SIZE - before_1 - shower_image.shape[0]
            before_2 = np.random.randint(self.cfg.IMAGE_SIZE - shower_image.shape[1])
            after_2 = self.cfg.IMAGE_SIZE - before_2 - shower_image.shape[1]
            #print(before_1, after_1, before_2, after_2)
            output_showers = output_showers + np.pad(shower_image, ((before_1, after_1), (before_2, after_2)), 'constant', constant_values=0)
            shower_start_points.append((int(shower_start_points_i[0]*scale) + before_1, int(shower_start_points_i[1]*scale) + before_2))
            angle.append(angle_i)

        return np.clip(output_showers, 0, 1), shower_start_points, angle

    def forward(self):
        track_length = 0.0
        kinks = 0
        if self.classification:
            if self.kinks is None:
                if np.random.uniform() < 0.5:
                    output_showers, shower_start_points = np.zeros((self.N, self.N)), []
                    output_tracks, track_start_points, track_end_points = generate_toy_tracks(self.N, self.max_tracks,
                                                                                            max_kinks=self.max_kinks,
                                                                                            max_track_length=self.max_track_length,
                                                                                            padding=self.gt_box_padding)
                    # start and end are ill-defined without charge gradient
                    track_edges = track_start_points + track_end_points
                    image_label = 1 # Track image
                    kinks = len(track_start_points)
                    track_length = np.sqrt(np.power(track_start_points[0][0]-track_end_points[0][0], 2) + np.power(track_start_points[0][1] - track_end_points[0][1], 2))
                else:
                    output_showers, shower_start_points, angle = self.make_showers()
                    output_tracks = np.zeros((self.N, self.N))
                    track_edges = []
                    image_label = 2 # shower image
            else:
                output_showers, shower_start_points = np.zeros((self.N, self.N)), []
                output_tracks, track_start_points, track_end_points = generate_toy_tracks(self.N, self.max_tracks,
                                                                                        max_kinks=self.max_kinks,
                                                                                        max_track_length=self.max_track_length,
                                                                                        padding=self.gt_box_padding,
                                                                                        kinks=self.kinks)
                # start and end are ill-defined without charge gradient
                track_edges = track_start_points + track_end_points
                image_label = 1 # Track image
                kinks = len(track_start_points)
                track_length = np.sqrt(np.power(track_start_points[0][0]-track_end_points[0][0], 2) + np.power(track_start_points[0][1] - track_end_points[0][1], 2))

        else:
            output_showers, shower_start_points, angle = self.make_showers()
            output_tracks, track_start_points, track_end_points = generate_toy_tracks(self.N, self.max_tracks, max_kinks=self.max_kinks, max_track_length=self.max_track_length, padding=self.gt_box_padding)
            # start and end are ill-defined without charge gradient
            track_edges = track_start_points + track_end_points

        bbox_labels = []
        simple_labels = []
        gt_pixels = []

        # find bbox for shower
        # FIXME what happens if output_showers is empty ?
        if shower_start_points:
            for i in range(len(shower_start_points)):
                bbox_labels.append([shower_start_points[i][0]-self.gt_box_padding,
                                    shower_start_points[i][1]-self.gt_box_padding,
                                    shower_start_points[i][0]+self.gt_box_padding,
                                    shower_start_points[i][1]+self.gt_box_padding,
                                    2]) # 2 for shower_start
                simple_labels.append([2])
                gt_pixels.append([shower_start_points[i][0], shower_start_points[i][1], 2])
            simple_label = 2
            opening_angle = angle

        # find bbox for tracks
        if track_edges:
            for i in range(len(track_edges)):
                bbox_labels.append([track_edges[i][0]-self.gt_box_padding,
                                    track_edges[i][1]-self.gt_box_padding,
                                    track_edges[i][0]+self.gt_box_padding,
                                    track_edges[i][1]+self.gt_box_padding,
                                    1 # 1 for track_edge
                             ])
                simple_labels.append([1])
                gt_pixels.append([track_edges[i][0], track_edges[i][1], 1])
            simple_label = 1
            opening_angle = None

        output = np.maximum(output_showers, output_tracks).reshape([1, self.N, self.N, 1])

        #output = np.repeat(output, 3, axis=3) # FIXME VGG needs RGB channels?

        blob = {}
        #img = np.concatenate([img,img,img],axis=3)
        blob['data'] = output.astype(np.float32)
        blob['im_info'] = [1, self.N, self.N, 3]
        blob['gt_boxes'] = np.array(bbox_labels)
        # Ji Won
        blob['class_labels'] = np.array([[simple_label]])
        blob['angles'] = np.array([opening_angle])
        # Laura
        blob['gt_labels'] = np.array(simple_labels)
        blob['gt_pixels'] = np.array(gt_pixels)
        if self.classification:
            blob['image_label'] = np.array([[image_label]])
        if self.classification and image_label == 1:
            blob['track_length'] = track_length
            blob['kinks'] = kinks

        return blob

    def fetch_batch(self):
        batch_blob = [self.forward() for i in range(self.batch_size)]
        batch_data = np.concatenate([d['data'] for d in batch_blob], axis=0)
        batch_labels = np.concatenate([d['class_labels'] for d in batch_blob], axis=0).reshape(-1)
        batch_angles = np.concatenate([d['angles'] for d in batch_blob], axis=0).reshape(-1)

        blob = {}
        blob['data'] = batch_data
        blob['class_labels'] = batch_labels
        blob['angles'] = batch_angles
        return blob

if __name__ == '__main__':
    t = ToydataGenerator(256, 3, 1, batch_size=20, classification=False)
    blobdict = t.forward()
    print(blobdict['gt_boxes'])
    print(blobdict['data'].shape)
    print(blobdict['class_labels'].shape)
    print("gt pixels shape ", blobdict['gt_pixels'].shape)

    #b = t.fetch_batch()
    #print(b['data'].shape)
    #print(b['class_labels'].shape)
    #print(b['angles'].shape)
