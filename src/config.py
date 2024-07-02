from yacs.config import CfgNode as CN

cfg = CN()

cfg.DATA_ROOT = "/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR/TD-TSR"
cfg.MODEL = 'faster_rcnn'
cfg.NUM_CLASSES = 6
cfg.BATCH_SIZE = 4
cfg.EPOCHS = 1
cfg.BASE_LR = 0.004
cfg.WEIGHT_DECAY = 0.0005
cfg.NO_CUDA = False
cfg.RUN_NAME = "run2"
cfg.DO_EARLY_STOPPING = False
cfg.BACKBONE = "resnet18"
cfg.SAVE_MODEL = True
def get_cfg_defaults():
    return cfg.clone()
