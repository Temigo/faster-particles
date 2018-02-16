# *-* encoding: utf-8 *-*
# Generate toy dataset
# with labels = feature space points (track and shower start/end points)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
from track_generator import generate_toy_tracks
from shower_generator import make_shower

class ToydataGenerator(object):
    CLASSES = ('__background__', 'track_start', 'track_end', 'shower_start')

    def __init__(self, N, max_tracks, max_kinks):
        self.N = N # shape of canvas
        self.max_tracks = max_tracks
        self.max_kinks = max_kinks
        self.args_def = dict(
            nx = N,
            ny = N,
            nlines = 10,
            dtheta = np.radians(20),
            lmin = 40,
            lmax = 127,
            keep = 7,
            keep_prob = 0.6,
            nimages = 2,
            out_png = False,
        )
        
    def forward(self):
        output_showers, shower_start_points = make_shower(self.args_def)
        output_tracks, track_start_points, track_end_points = generate_toy_tracks(self.N, self.max_tracks, max_kinks=self.max_kinks)
        # start and end are ill-defined without charge gradient
        track_edges = track_start_points + track_end_points

        bbox_labels = []

        # find bbox for shower
        rmin, rmax = np.where(np.any(output_showers, axis=1))[0][[0, -1]]
        cmin, cmax = np.where(np.any(output_showers, axis=0))[0][[0, -1]]
        bbox_labels.append([1, rmin-2, rmax+2, cmin-2, cmax+2, 0]) # 0 for shower_start

        # find bbox for tracks
        for i in range(len(track_edges)):
            bbox_labels.append([1, 
                                 track_edges[i][0]-2,
                                 track_edges[i][1]-2,
                                 track_edges[i][0]+2,
                                 track_edges[i][1]+2,
                                 1 # 1 for track_edge
                             ])

        output = np.maximum(output_showers, output_tracks).reshape([1, self.N, self.N, 1])

        blob = {}
        #img = np.concatenate([img,img,img],axis=3)
        blob['data'] = output
        blob['im_info'] = [1, self.N, self.N, 1]
        blob['gt_boxes'] = np.array(bbox_labels)

        return blob

if __name__ == '__main__':
    t = ToydataGenerator(256, 3, 1)
    blobdict = t.forward()
    print blobdict['gt_boxes']
    print blobdict['data'].shape
    
