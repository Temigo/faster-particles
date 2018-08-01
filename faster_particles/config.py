import argparse
import os
import numpy as np
import tensorflow as tf


from demo_ppn import inference, inference_full, inference_ppn_ext
from train_ppn import train_ppn, train_classification, train_small_uresnet



os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# Control Tensorflow verbose level with TF_CPP_MIN_LOG_LEVEL
# it defaults to 0 (all logs shown), but can be set to 1 to filter out INFO logs,
# 2 to additionally filter out WARNING logs, and 3 to additionally filter out ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class PPNConfig(object):
    IMAGE_SIZE = 768 # 512
    CROP_SIZE = 24
    OUTPUT_DIR = "output"
    LOG_DIR = "log"
    DISPLAY_DIR = "display"
    NUM_CLASSES = 3
    R = 20
    PPN1_SCORE_THRESHOLD = 0.6
    PPN2_DISTANCE_THRESHOLD = 5
    LEARNING_RATE = 0.001
    LAMBDA_PPN = 0.5
    LAMBDA_PPN1 = 0.5
    LAMBDA_PPN2 = 0.5
    WEIGHTS_FILE = None # Path to pretrained checkpoint
    WEIGHTS_FILE_BASE = None
    WEIGHTS_FILE_PPN = None
    WEIGHTS_FILE_SMALL = None
    FREEZE = False # Whether to freeze the weights of base net
    NET = 'ppn'
    BASE_NET = 'vgg'
    MAX_STEPS = 100
    WEIGHT_LOSS = False # FIXME make it False by default
    MIN_SCORE=0.0

    # Data configuration
    BATCH_SIZE = 1
    SEED = 123
    NEXT_INDEX = 0
    TOYDATA = False
    # DATA = "/data/drinkingkazu/dlprod_ppn_v05/ppn_p[01]*.root"
    # DATA = "/stage/drinkingkazu/dlprod_ppn_v05/ppn_p01.root" # For 2D
    # DATA = "/data/drinkingkazu/dlprod_ppn_v05/ppn_p00.root" # For 3D
    # DATA = "/data/drinkingkazu/dlprod_ppn_v05/ppn_p00_0000_0019.root" # For 3D
    # DATA = "/data/drinkingkazu/dlprod_multipvtx_v05/mix/hit_mix00.root" # For UResNet 2D
    #DATA = "/stage/drinkingkazu/u-resnet/vertex_data/out.root" # For UResNet 3D
    # or /stage/drinkingkazu/u-resnet/multipvtx_data/out.root
    # DATA = ""
    #DATA = "/stage/drinkingkazu/dlprod_ppn_v06/train.root"
    #DATA = "/stage/drinkingkazu/dlprod_ppn_v06/blur_train.root"
    DATA = "/stage/drinkingkazu/fuckgrid/p*/larcv.root"
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

    # Environment variables
    CUDA_VISIBLE_DEVICES = '0,1,2,3'

    def __init__(self):
        self.create_parsers()
        os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA_VISIBLE_DEVICES

    def create_parsers(self):
        self.parser = argparse.ArgumentParser(description="Pixel Proposal Network")
        subparsers = self.parser.add_subparsers(title="Modules", description="Valid subcommands", dest='script', help="Train or run PPN")

        self.train_parser = subparsers.add_parser("train", help="Train Pixel Proposal Network")
        self.train_parser.add_argument("-o", "--output-dir", action='store', type=str, required=True, help="Path to output directory.")
        self.train_parser.add_argument("-l", "--log-dir", action='store', type=str, required=True, help="Path to log directory.")
        self.train_parser.add_argument("-c", "--num-classes", default=self.NUM_CLASSES, type=int, help="Number of classes (including background).")
        self.train_parser.add_argument("-r", "--r", default=self.R, type=int, help="Max number of ROIs from PPN1")
        self.train_parser.add_argument("-st", "--ppn1-score-threshold", default=self.PPN1_SCORE_THRESHOLD, type=float, help="Threshold on signal score to define positives in PPN1")
        self.train_parser.add_argument("-dt", "--ppn2-distance-threshold", default=self.PPN2_DISTANCE_THRESHOLD, type=float, help="Threshold on distance to closest ground truth pixel to define positives in PPN2")
        self.train_parser.add_argument("-lr", "--learning-rate", default=self.LEARNING_RATE, type=float, help="Learning rate")
        self.train_parser.add_argument("-lppn", "--lambda-ppn", default=self.LAMBDA_PPN, type=float, help="Lambda PPN (for loss weighting)")
        self.train_parser.add_argument("-lppn1", "--lambda-ppn1", default=self.LAMBDA_PPN1, type=float, help="Lambda PPN1")
        self.train_parser.add_argument("-lppn2", "--lambda-ppn2", default=self.LAMBDA_PPN2, type=float, help="Lambda PPN2")
        self.train_parser.add_argument("-wl", "--weight-loss", default=self.WEIGHT_LOSS, action='store_true', help="Weight the loss (balance track and shower)")
        self.train_parser.add_argument("-f", "--freeze", default=self.FREEZE, action='store_true', help="Freeze the base net weights.")

        self.demo_parser = subparsers.add_parser("demo", help="Run Pixel Proposal Network demo.")
        self.demo_full_parser = subparsers.add_parser("demo-full", help="Run Pixel Proposal Network combined with base UResNet demo.")

        self.common_arguments(self.train_parser)
        self.common_arguments(self.demo_parser)
        self.common_arguments(self.demo_full_parser)

        self.demo_parser.set_defaults(func=inference)
        self.demo_full_parser.set_defaults(func=inference_full)
        self.train_parser.set_defaults(func=train_ppn)

    def common_arguments(self, parser):
        parser.add_argument("-m", "--max-steps", default=self.MAX_STEPS, type=int, help="Maximum number of training iterations.")
        parser.add_argument("-wb", "--weights-file-base", help="Tensorflow .ckpt file to load weights of trained base network.")
        parser.add_argument("-wp", "--weights-file-ppn", help="Tensorflow .ckpt file to load weights of trained PPN.") # does not load base net weights
        parser.add_argument("-ws", "--weights-file-small", help="Tensorflow .ckpt file to load weights of small UResNet.")
        parser.add_argument("-gpu", "--gpu", default=self.CUDA_VISIBLE_DEVICES, type=str, help="CUDA visible devices list (in a string and separated by commas).")
        parser.add_argument("-3d", "--data-3d", default=self.DATA_3D, action='store_true', help="Use 3D instead of 2D.")
        parser.add_argument("-data", "--data", default=self.DATA, type=str, help="Path to data files. Can use ls regex format.")
        parser.add_argument("-td", "--toydata", default=self.TOYDATA, action='store_true', help="Whether to use toydata or not")
        parser.add_argument("-bn", "--base-net", default=self.BASE_NET, type=str, help="Base network of PPN (e.g. VGG)")
        parser.add_argument("-n", "--net", default=self.NET, type=str, choices=['ppn', 'base', 'full', 'small_uresnet', 'ppn_ext'], help="Whether to use base net or PPN net or both.")
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
        parser.add_argument("-d", "--display-dir", action='store', type=str, required=True, help="Path to display directory.")
        parser.add_argument("-ni", "--next-index", default=self.NEXT_INDEX, type=int, help="Index from which to start reading LArCV data file.")

    def parse_args(self):
        args = self.parser.parse_args()
        self.update(vars(args))
        print("\n\n-- CONFIG --")
        for v in vars(self):
            print("%s = %r" % (v, getattr(self, v)))
        print("\n\n")
        if self.NET == 'base' and args.script == 'train':
            args.func = train_classification
        elif self.NET == 'small_uresnet' and args.script == 'train':
            args.func = train_small_uresnet
        elif self.NET == 'ppn_ext' and args.script == 'demo':
            args.func = inference_ppn_ext
        if self.FREEZE and self.WEIGHTS_FILE_BASE is None:
            print("WARNING You are freezing the base net weights without loading any checkpoint file.")

        # Set random seed for reproducibility
        np.random.seed(self.SEED)
        tf.set_random_seed(self.SEED)

        args.func(self)

    def update(self, args):
        for name in args:
            if name != "func" and name != 'script':
                setattr(self, name.upper(), args[name])

cfg = PPNConfig()
