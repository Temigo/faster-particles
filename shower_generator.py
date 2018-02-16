import numpy as np
from skimage.draw import line_aa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
#%matplotlib inline

args_def = dict(
    nx = 128,        
    ny = 128,      
    nlines = 10,     
    dtheta = np.radians(20),        
    lmin = 20,
    lmax = 63,
    keep = 7,
    keep_prob = 0.6,
    N = 2,
    out_png = True,
)

def make_shower(args, nx = 128, ny = 128):

    img = np.zeros(shape=(nx, ny), dtype=int)

    # randomly generate starting point
    # note there is a lmax buffer on the boundaries of canvas
    # so that shower doesn't fall off the image
    vx, vy = np.random.randint(args.lmax,nx-args.lmax), np.random.randint(args.lmax,ny-args.lmax)
    theta0 = np.random.uniform(2.*np.pi) # central angle of shower

    # randomly generate nlines endpoints such that the lines fall
    # within around dtheta of theta0
    thetas = np.random.normal(loc=theta0, scale=args.dtheta, size=(args.nlines,1))
    lengths = np.random.uniform(low=args.lmin, high=args.lmax, size=(args.nlines,1))

    # draw shower lines
    for pos in np.hstack(((vx+lengths*np.cos(thetas)+0.5).astype(int), (vy+lengths*np.sin(thetas)+0.5).astype(int))):
        rr, cc, _ = line_aa(vx, vy, pos[0], pos[1])
        img[rr, cc] = 1

    # randomly set pixels to 0
    indices0 = np.random.choice([0, 1], p=[args.keep_prob, 1-args.keep_prob], size=img.shape).astype(np.bool)
    #indices0 = np.random.randint(0,2,size=img.shape).astype(np.bool)
    indices1 = np.ones(img.shape)
    indices1[vx-args.keep:vx+args.keep, vy-args.keep:vy+args.keep] = 0
    img[np.logical_and(indices0, indices1)] = 0

    return img, (vx,vy)

def make_showerset(args):
    for i in range(args.N):
        img = make_shower(args)
        np.savetxt('shower_%d.txt'%i,img)
        if args.out_png:
            plt.imshow(img)
            plt.savefig('shower_%d.png'%i)
            plt.close()
        if i != 0 and i%20==0: print(i, ' done')        

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    
    parser.add_argument('--nx', type=int, default = args_def['nx'],
                        help='x dimension of canvas, 128')
    
    parser.add_argument("--ny", type=int, default = args_def['ny'],
                        help="y dimension of canvas, 128")
    
    parser.add_argument("--nlines", type=int, default = args_def['nlines'],
                        help="number of shower lines coming out of starting point, 10")
    
    parser.add_argument("--dtheta", type=int, default = args_def['dtheta'],
                        help="angular range of shower lines (stdev of Gaussian), 20 deg")
    
    parser.add_argument("--lmin", type=int, default = args_def['lmin'],
                        help="minimum shower line length: must be < lmax, 20")
    
    parser.add_argument("--lmax", type=int, default = args_def['lmax'],
                        help="maximum shower line length: must be in (lmin+1, max(nx,ny)/2-1), 63")
        
    parser.add_argument("--keep", type=int, default = args_def['keep'],
                        help="number of pixels around the starting point to keep, 7")
    
    parser.add_argument('--keep_prob', type=float, default = args_def['keep_prob'],
                        help='probability of keeping a pixel, 0.6')

    parser.add_argument('--N', type=int, default = args_def['N'],
                        help='number of images to generate, 100')

    parser.add_argument('--out_png', type=bool, default=args_def['out_png'],
                        help='whether to output png file, False')
    
    args = parser.parse_args()
    make_showerset(args)
