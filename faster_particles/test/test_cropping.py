from faster_particles.larcvdata.larcvdata_generator import LarcvGenerator
from faster_particles import cropping
from faster_particles.display_utils import display_uresnet


class MyCfg:
    IMAGE_SIZE = 512
    SEED = 123
    BATCH_SIZE = 1
    DATA_3D = True
    NET = "base"
    BASE_NET = "uresnet"
    NEXT_INDEX = 0
    CROP_SIZE = 24
    SLICE_SIZE = 128
    DISPLAY_DIR = "display/train_uresnet3d"


cfg = MyCfg()
t = LarcvGenerator(cfg, ioname='test',
                   filelist='["/data/dlprod_ppn_v08_p01/train.root"]')
crop_algorithm = cropping.Probabilistic(cfg)
for i in range(1):
    print(i)
    blob = t.forward()
    display_uresnet(blob, cfg, directory=cfg.DISPLAY_DIR, index=0)
    batch_blobs = crop_algorithm.process(blob)
    cfg.IMAGE_SIZE = cfg.SLICE_SIZE
    for j, b in enumerate(batch_blobs):
        print("\t %d" % j)
        print(b['data'].shape, b['labels'].shape)
        display_uresnet(b, cfg, directory=cfg.DISPLAY_DIR, index=1000*i+j+1)
