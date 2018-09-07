# *-* encoding: utf-8 *-*
# Training for PPN, base network and small UResNet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from faster_particles.ppn import PPN
from faster_particles.demo_ppn import get_data
from faster_particles.display_utils import display, display_uresnet, \
                                            display_ppn_uresnet
from faster_particles.base_net import basenets
from faster_particles.trainer import Trainer


def train(cfg):
    """
    Launch PPN training with appropriate dataset and base layers.
    """
    batch_size = cfg.BATCH_SIZE
    if cfg.ENABLE_CROP:
        cfg.BATCH_SIZE = 1

    train_data, test_data = get_data(cfg)
    net_args = {"base_net": basenets[cfg.BASE_NET], "base_net_args": {}}
    if cfg.NET == "base":
        net_args = {}
    if cfg.NET == "small_uresnet":
        net_args = {"N": cfg.CROP_SIZE}

    display_util = display_ppn_uresnet
    if cfg.NET == "ppn":
        display_util = display
    elif cfg.NET == "base":
        if cfg.BASE_NET == "uresnet":
            display_util = display_uresnet
        else:
            display_util = None
    elif cfg.NET == "small_uresnet":
        display_util = display_uresnet

    net = PPN if cfg.NET == "ppn" else basenets[cfg.BASE_NET]

    scope = "ppn"
    if cfg.NET == "base":
        scope = cfg.BASE_NET
    elif cfg.NET == "small_uresnet":
        scope = "small_uresnet"

    cfg.BATCH_SIZE = batch_size
    t = Trainer(net,
                train_data,
                test_data,
                cfg,
                display_util=display_util)
    t.train(net_args, scope=scope)
