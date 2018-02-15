# *-* encoding: utf-8 *-*
# Generate tracks for toy data
# Usage : python track_generator.py N max_tracks boolCsv nb-images
# or
# from track_generator import generate_toy_tracks
# output = generate_toy_tracks(N, max_tracks [, filename, output])

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import sys

def draw_line(output, p1, p2):
    # Modify output in place
    # Order start and end by x value
    start = p1
    end = p2
    low = True
    if abs(p2[1] - p1[1]) < abs(p2[0] - p1[0]):
        if p1[0] > p2[0]:
            start = p2
            end = p1
    else:
        low = False
        if p1[1] > p2[1]:
            start = p2
            end = p1

    # Bresenham algorithm
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if low:
        yi = 1
        if dy < 0:
            yi = -1
            dy = -dy
        e = 2*dy - dx
        y = start[1]
        for x in range(start[0], end[0]+1):
            output[x, y] = 1
            if e > 0:
                y += yi
                e -= 2*dx
            e += 2*dy
    else:
        xi = 1
        if dx < 0:
            xi = -1
            dx = -dx
        e = 2*dx - dy
        x = start[0]
        for y in range(start[1], end[1]+1):
            output[x,y] = 1
            if e>0:
                x += xi
                e -= 2*dy
            e += 2*dx

def generate_toy_tracks(N, max_tracks, filename='', out_format='png'):
    nb_tracks = np.random.randint(low=1, high=max_tracks+1)
    output = np.zeros(shape=(N, N), dtype=int)

    print "\nGenerating %d x %d image with %d tracks (at most %d tracks)" % (N, N, nb_tracks, max_tracks)
    print "Save to %s\n" % out_format

    for i_track in range(nb_tracks):
        # Generate one track
        #length = np.random.uniform(high=np.sqrt(2.0) * N)
        length = np.random.uniform(high=N)
        theta = np.random.uniform(high=2.0*np.pi)
        start = (np.random.randint(low=0, high=N), np.random.randint(low=0, high=N))
        end = (np.clip(start[0] + int(length * np.cos(theta)), 0, 127), np.clip(start[1] + int(length * np.sin(theta)), 0, 127))
        print i_track, start, end
        draw_line(output, start, end)

    if len(filename):
        if out_format == 'csv':
            with open(filename, 'a') as f:
                np.savetxt(f, output, delimiter=",")
        else:
            plt.imsave(filename, output)#, cmap=cm.gray)
            print "\n%s saved.\n" % filename
    return output

if __name__ == '__main__':
    #generate_toy_tracks(128, 5)
    if len(sys.argv) < 4:
        print "Usage : python track_generator.py N max_tracks boolCsv nb-images"
    else:
        if int(sys.argv[3]):
            for _ in range(int(sys.argv[4])):
                generate_toy_tracks(int(sys.argv[1]), int(sys.argv[2]), filename='toy_tracks.csv', out_format='csv')
        else:
            for i in range(int(sys.argv[4])):
                generate_toy_tracks(int(sys.argv[1]), int(sys.argv[2]), filename='toy_tracks_%d.png' % i, out_format='png')
