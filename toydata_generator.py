# *-* encoding: utf-8 *-*
# Generate toy dataset
# with labels = feature space points (track and shower start/end points)

import numpy as np
import sys
from track_generator import generate_toy_tracks
# from shower_generator import *

class ToydataGenerator(object):
    def __init__(self, N, max_tracks, max_kinks):
        self.N = N
        self.max_tracks = max_tracks
        self.max_kinks = max_kinks

    def forward(self):
        output = np.zeros(shape=(self.N, self.N), dtype=int)
        output_tracks, track_start_points, track_end_points = generate_toy_tracks(self.N, self.max_tracks, max_kinks=self.max_kinks)
        # output_shower =
        blob['image'] = output
        # blob['gt'] =
        return blob

if __name__ == '__main__':
    t = ToydataGenerator(128, 5, 1)
    print t.forward()
