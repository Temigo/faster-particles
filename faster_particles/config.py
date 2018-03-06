import argparse
import os
from demo_ppn import inference
from train_ppn import train_ppn, train_classification

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class PPNConfig(object):
    IMAGE_SIZE = 512
    OUTPUT_DIR = "output"
    LOG_DIR = "log"
    DISPLAY_DIR = "display"
    NUM_CLASSES = 3
    MAX_TRACKS = 5
    MAX_KINKS = 2
    MAX_TRACK_LENGTH = 200
    R = 20
    PPN1_SCORE_THRESHOLD = 0.5
    PPN2_DISTANCE_THRESHOLD = 5
    LEARNING_RATE = 0.001
    LAMBDA_PPN = 0.5
    LAMBDA_PPN1 = 0.5
    LAMBDA_PPN2 = 0.5
    MAX_STEPS = 100
    KINKS = None
    BATCH_SIZE = 20
    DTHETA = -1
    SEED = 123
    WEIGHTS_FILE = None # Path to pretrained checkpoint
    NET = 'ppn'

    def __init__(self):
        self.create_parsers()

    def create_parsers(self):
        self.parser = argparse.ArgumentParser(description="Pixel Proposal Network")
        subparsers = self.parser.add_subparsers(title="Modules", description="Valid subcommands", help="Train or run PPN")

        self.train_parser = subparsers.add_parser("train", help="Train Pixel Proposal Network")
        self.train_parser.add_argument("-o", "--output-dir", action='store', type=str, required=True, help="Path to output directory.")
        self.train_parser.add_argument("-l", "--log-dir", action='store', type=str, required=True, help="Path to log directory.")
        self.train_parser.add_argument("-c", "--num-classes", default=self.NUM_CLASSES, type=int, help="Number of classes (including background).")
        self.train_parser.add_argument("-m", "--max-steps", default=self.MAX_STEPS, type=int, help="Maximum number of training iterations.")
        self.train_parser.add_argument("-n", "--net", default='ppn', type=str, choices=['ppn', 'base'], help="Whether to train base net or PPN net.")

        self.demo_parser = subparsers.add_parser("demo", help="Run Pixel Proposal Network demo.")
        self.demo_parser.add_argument("weights_file", help="Tensorflow .ckpt file to load weights of trained model.")

        self.common_arguments(self.train_parser)
        self.common_arguments(self.demo_parser)

        self.demo_parser.set_defaults(func=inference)
        self.train_parser.set_defaults(func=train_ppn)

    def common_arguments(self, parser):
        parser.add_argument("-N", "--image-size", action='store', default=self.IMAGE_SIZE, type=int, choices=[128, 256, 512], help="Width (and height) of image.")
        parser.add_argument("-d", "--display-dir", action='store', type=str, required=True, help="Path to display directory.")
        parser.add_argument("-t", "--max-tracks", default=self.MAX_TRACKS, type=int, help="Maximum number of tracks generated per image (uniform distribution).")
        parser.add_argument("-k", "--max-kinks", default=self.MAX_KINKS, type=int, help="Maximum number of kinks generated for any track.")
        parser.add_argument("-tl", "--max-track-length", default=self.MAX_TRACK_LENGTH, type=float, help="Maximum length of any track.")

    def parse_args(self):
        args = self.parser.parse_args()
        self.update(vars(args))
        if self.NET == 'base':
            args.func = train_classification
        print self.NUM_CLASSES
        args.func(self)

    def update(self, args):
        for name in args:
            if name != "func":
                setattr(self, name.upper(), args[name])

cfg = PPNConfig()
