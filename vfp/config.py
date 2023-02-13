import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# pair size for trainning set
_C.DATA.PAIR_SIZE = 1
_C.DATA.TRAIN_LABEL = './dataset/train_eval/train_label.npy'
_C.DATA.TRAIN_FEAT = './dataset/train_eval/train_feat_RN101.npy'
_C.DATA.QUERY_FEAT = './dataset/train_eval/val_img_feat_RN101.npy'
_C.DATA.GALLERY_FEAT = './dataset/train_eval/val_txt_feat_RN101.npy'

# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
_C.MODEL.RESUME_KEY = 'model'

# -----------------------------------------------------------------------------
# Binary settings
# -----------------------------------------------------------------------------
_C.BINARY = CN()
_C.BINARY.ACTIVATE = False

_C.BINARY.TRAIN = CN()
_C.BINARY.TRAIN.PROXY_LOSS = False

_C.BINARY.TRANS = CN()
_C.BINARY.TRANS.TYPE = 'rbe'

# <Recurrent Binary Embedding for GPU-Enabled Exhaustive Retrieval from Billion-Scale Semantic Vectors>
_C.BINARY.TRANS.RBE = CN()
_C.BINARY.TRANS.RBE.INPUT_DIM = 64
_C.BINARY.TRANS.RBE.OUTPUT_DIM = 128
_C.BINARY.TRANS.RBE.NUM_LAYERS = 1
_C.BINARY.TRANS.RBE.TRANSFORM_BLOCKS = 1
_C.BINARY.TRANS.RBE.HIDDEN_DIM = 0
_C.BINARY.TRANS.RBE.BIAS = False
_C.BINARY.TRANS.RBE.BINARY_FUNC = 'st_var'
_C.BINARY.TRANS.RBE.ANNEAL_STEP = 1


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = 'pairwise_contrastive'
_C.TRAIN.LOSS.TYPE = 'contra' # or triplet
_C.TRAIN.LOSS.TEMP = 0.1 # for contrastive loss
_C.TRAIN.LOSS.MARGIN = 0.8 # for triplet loss
_C.TRAIN.LOSS.RM_DUP = True # remove negatives of the same vid
_C.TRAIN.LOSS.HARD_TOPK = 5 # None for no hard mining
_C.TRAIN.LOSS.QUEUE = 0 # expand the selection pool

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
# Path to output folder, overwritten by command line argument
_C.OUTPUT = 'output'
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 10
_C.EVAL_FREQ = 10
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.resume_key:
        config.MODEL.RESUME_KEY = args.resume_key
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
