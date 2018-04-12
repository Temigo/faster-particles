import argparse
import os
from demo_ppn import inference
from train_ppn import train_ppn, train_classification

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Control Tensorflow verbose level with TF_CPP_MIN_LOG_LEVEL
# it defaults to 0 (all logs shown), but can be set to 1 to filter out INFO logs,
# 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class PPNConfig(object):
    IMAGE_SIZE = 768 # 512
    OUTPUT_DIR = "output"
    LOG_DIR = "log"
    DISPLAY_DIR = "display"
    NUM_CLASSES = 3
    R = 20
    PPN1_SCORE_THRESHOLD = 0.5
    PPN2_DISTANCE_THRESHOLD = 5
    LEARNING_RATE = 0.001
    LAMBDA_PPN = 0.5
    LAMBDA_PPN1 = 0.5
    LAMBDA_PPN2 = 0.5
    WEIGHTS_FILE = None # Path to pretrained checkpoint
    NET = 'ppn'
    BASE_NET = 'vgg'
    MAX_STEPS = 100
    WEIGHT_LOSS = True # FIXME make it False by default
    MIN_SCORE=0.0

    # Data configuration
    BATCH_SIZE = 1
    SEED = 123
    TOYDATA = False
    # DATA = "/data/drinkingkazu/dlprod_ppn_v05/ppn_p[01]*.root"
    DATA = "/stage/drinkingkazu/dlprod_ppn_v05/ppn_p01.root" # For 2D
    # DATA = "/data/drinkingkazu/dlprod_ppn_v05/ppn_p00_0000_0019.root" # For 3D
    # DATA = ""
    DATA_3D = False

    # Track configuration
    MAX_TRACKS = 5
    MAX_KINKS = 2
    MAX_TRACK_LENGTH = 200
    KINKS = None

    # Shower configuration
    MAX_SHOWERS = 5
    SHOWER_N_LINES = 10
    SHOWER_DTHETA = -1
    SHOWER_L_MIN = 40
    SHOWER_L_MAX = 127
    SHOWER_KEEP = 7
    SHOWER_KEEP_PROB = 0.6
    SHOWER_N_IMAGES = 2
    SHOWER_OUT_PNG = False

    def __init__(self):
        self.create_parsers()

    def create_parsers(self):
        self.parser = argparse.ArgumentParser(description="Pixel Proposal Network")
        subparsers = self.parser.add_subparsers(title="Modules", description="Valid subcommands", dest='script', help="Train or run PPN")

        self.train_parser = subparsers.add_parser("train", help="Train Pixel Proposal Network")
        self.train_parser.add_argument("-o", "--output-dir", action='store', type=str, required=True, help="Path to output directory.")
        self.train_parser.add_argument("-l", "--log-dir", action='store', type=str, required=True, help="Path to log directory.")
        self.train_parser.add_argument("-d", "--display-dir", action='store', type=str, required=True, help="Path to display directory.")
        self.train_parser.add_argument("-c", "--num-classes", default=self.NUM_CLASSES, type=int, help="Number of classes (including background).")
        self.train_parser.add_argument("-m", "--max-steps", default=self.MAX_STEPS, type=int, help="Maximum number of training iterations.")
        self.train_parser.add_argument("-r", "--r", default=self.R, type=int, help="Max number of ROIs from PPN1")
        self.train_parser.add_argument("-st", "--ppn1-score-threshold", default=self.PPN1_SCORE_THRESHOLD, type=float, help="Threshold on signal score to define positives in PPN1")
        self.train_parser.add_argument("-dt", "--ppn2-distance-threshold", default=self.PPN2_DISTANCE_THRESHOLD, type=float, help="Threshold on distance to closest ground truth pixel to define positives in PPN2")
        self.train_parser.add_argument("-lr", "--learning-rate", default=self.LEARNING_RATE, type=float, help="Learning rate")
        self.train_parser.add_argument("-lppn", "--lambda-ppn", default=self.LAMBDA_PPN, type=float, help="Lambda PPN (for loss weighting)")
        self.train_parser.add_argument("-lppn1", "--lambda-ppn1", default=self.LAMBDA_PPN1, type=float, help="Lambda PPN1")
        self.train_parser.add_argument("-lppn2", "--lambda-ppn2", default=self.LAMBDA_PPN2, type=float, help="Lambda PPN2")
        self.train_parser.add_argument("-w", "--weights-file", help="Tensorflow .ckpt file to load weights of trained model.")
        self.train_parser.add_argument("-wl", "--weight-loss", default=self.WEIGHT_LOSS, action='store_true', help="Weight the loss (balance track and shower)")

        self.demo_parser = subparsers.add_parser("demo", help="Run Pixel Proposal Network demo.")
        self.demo_parser.add_argument("weights_file", help="Tensorflow .ckpt file to load weights of trained model.")
        self.demo_parser.add_argument("-d", "--display-dir", action='store', type=str, required=True, help="Path to display directory.")

        self.common_arguments(self.train_parser)
        self.common_arguments(self.demo_parser)

        self.demo_parser.set_defaults(func=inference)
        self.train_parser.set_defaults(func=train_ppn)

    def common_arguments(self, parser):
        parser.add_argument("-3d", "--data-3d", default=self.DATA_3D, action='store_true', help="Use 3D instead of 2D.")
        parser.add_argument("-data", "--data", default=self.DATA, type=str, help="Path to data files. Can use ls regex format.")
        parser.add_argument("-td", "--toydata", default=self.TOYDATA, action='store_true', help="Whether to use toydata or not")
        parser.add_argument("-bn", "--base-net", default=self.BASE_NET, type=str, help="Base network of PPN (e.g. VGG)")
        parser.add_argument("-n", "--net", default=self.NET, type=str, choices=['ppn', 'base'], help="Whether to train base net or PPN net.")
        parser.add_argument("-N", "--image-size", action='store', default=self.IMAGE_SIZE, type=int, help="Width (and height) of image.")
        parser.add_argument("-mt", "--max-tracks", default=self.MAX_TRACKS, type=int, help="Maximum number of tracks generated per image (uniform distribution).")
        parser.add_argument("-mk", "--max-kinks", default=self.MAX_KINKS, type=int, help="Maximum number of kinks generated for any track.")
        parser.add_argument("-mtl", "--max-track-length", default=self.MAX_TRACK_LENGTH, type=float, help="Maximum length of any track.")
        parser.add_argument("-b", "--batch-size", default=self.BATCH_SIZE, type=int, help="Batch size")
        parser.add_argument("-s", "--seed", default=self.SEED, type=int, help="Random seed")
        parser.add_argument("-k", "--kinks", default=self.KINKS, type=int, help="Exact number of kinks to be generated for every track.")
        parser.add_argument("-maxs", "--max-showers", default=self.MAX_SHOWERS, type=int, help="Maximum number of showers generated per image (uniform distribution).")
        parser.add_argument("-snl", "--shower-n-lines", default=self.SHOWER_N_LINES, type=int, help="Number of lines generated per shower.")
        parser.add_argument("-sdt", "--shower-dtheta", default=self.SHOWER_DTHETA, type=float, help="Dtheta for shower generation.")
        parser.add_argument("-slmin", "--shower-l-min", default=self.SHOWER_L_MIN, type=float, help="Minimum length of line for shower generation.")
        parser.add_argument("-slmax", "--shower-l-max", default=self.SHOWER_L_MAX, type=float, help="Maximum length of line for shower generation.")
        parser.add_argument("-keep", "--keep", default=self.SHOWER_KEEP, type=float, help="")
        parser.add_argument("-kp", "--keep-prob", default=self.SHOWER_KEEP_PROB, type=float, help="")
        parser.add_argument("-nimages", "--shower-n-images", default=self.SHOWER_N_IMAGES, type=int, help="")
        parser.add_argument("-png", "--shower-out-png", default=self.SHOWER_OUT_PNG, action='store_true')
        parser.add_argument("-ms", "--min-score", default=self.MIN_SCORE, type=float, help="Minimum score above which PPN predictions should be kept")

    def parse_args(self):
        args = self.parser.parse_args()
        self.update(vars(args))
        if self.NET == 'base' and args.script == 'train':
            args.func = train_classification
        print self.NUM_CLASSES
        args.func(self)

    def update(self, args):
        for name in args:
            if name != "func" and name != 'script':
                setattr(self, name.upper(), args[name])

cfg = PPNConfig()
