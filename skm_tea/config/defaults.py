from meddlr.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# ==============================================================================
# Wrapper configs for blocks that are implemented multiple times.

# ---------------------------------------------------------------------------- #
# MONAI nets
# ---------------------------------------------------------------------------- #
_VNET_MONAI = CN()
_VNET_MONAI.ACTIVATION = ("elu", {"inplace": True})
_VNET_MONAI.DROPOUT_PROB = 0.5
_VNET_MONAI.DROPOUT_DIM = 2

_DYNUNET_MONAI = CN()
_DYNUNET_MONAI.KERNEL_SIZE = (3,)
_DYNUNET_MONAI.UPSAMPLE_KERNEL_SIZE = (2,)
_DYNUNET_MONAI.STRIDES = (1,)
_DYNUNET_MONAI.NORM_NAME = "instance"
_DYNUNET_MONAI.DEEP_SUPERVISION = False
_DYNUNET_MONAI.DEEP_SUPR_NUM = 1
_DYNUNET_MONAI.RES_BLOCK = False

_UNET_MONAI = CN()
_UNET_MONAI.CHANNELS = ()
_UNET_MONAI.STRIDES = ()
_UNET_MONAI.KERNEL_SIZE = (3,)
_UNET_MONAI.UP_KERNEL_SIZE = (3,)
_UNET_MONAI.NUM_RES_UNITS = 0
_UNET_MONAI.ACTIVATION = ("prelu", {})
_UNET_MONAI.NORM = ("instance", {})
_UNET_MONAI.DROPOUT = 0.0

_TB_RECON = CN()
_TB_RECON.CHANNELS = (16, 32, 64)
_TB_RECON.STRIDES = (2,)
_TB_RECON.KERNEL_SIZE = (5,)
_TB_RECON.ORDER = ("conv", "relu")
_TB_RECON.ENC_NUM_CONV_BLOCKS = (1, 2, 3)
_TB_RECON.DEC_NUM_CONV_BLOCKS = (2, 3)
_TB_RECON.MULTI_CONCAT = ()
_TB_RECON.USE_MAGNITUDE = False

# ==============================================================================


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.VERSION = 1

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.WEIGHTS = ""
_C.MODEL.TASKS = ("recon",)

# ---------------------
# General model params
_C.MODEL.PARAMETERS = CN()
_C.MODEL.PARAMETERS.USE_COMPLEX_WEIGHTS = False
# How to initialize model parameters. Should not be specified if INIT_WEIGHTS is specified.
# See skm_tea.modeling.meta_arch.build.initialize_model for more information.
_C.MODEL.PARAMETERS.INIT = ()

# -----------------------------------------------------------------------------
# Unrolled model
# -----------------------------------------------------------------------------
_C.MODEL.UNROLLED = CN()
_C.MODEL.UNROLLED.BLOCK_ARCHITECTURE = "ResNet"
_C.MODEL.UNROLLED.NUM_UNROLLED_STEPS = 5
_C.MODEL.UNROLLED.NUM_RESBLOCKS = 2
_C.MODEL.UNROLLED.NUM_FEATURES = 256
_C.MODEL.UNROLLED.DROPOUT = 0.0
# Padding options. "" for now. TODO: add "circular"
_C.MODEL.UNROLLED.PADDING = ""
_C.MODEL.UNROLLED.STEP_SIZES = (-2.0,)
_C.MODEL.UNROLLED.FIX_STEP_SIZE = False
_C.MODEL.UNROLLED.SHARE_WEIGHTS = False
# Kernel size
_C.MODEL.UNROLLED.KERNEL_SIZE = (3,)
# Number of ESPIRiT maps
_C.MODEL.UNROLLED.NUM_EMAPS = 1

# Conv block parameters
_C.MODEL.UNROLLED.CONV_BLOCK = CN()
# Either "relu" or "leaky_relu"
_C.MODEL.UNROLLED.CONV_BLOCK.ACTIVATION = "relu"
# Either "none", "instance", or "batch"
_C.MODEL.UNROLLED.CONV_BLOCK.NORM = "none"
# Use affine on norm
_C.MODEL.UNROLLED.CONV_BLOCK.NORM_AFFINE = False
_C.MODEL.UNROLLED.CONV_BLOCK.ORDER = ("norm", "act", "drop", "conv")

_C.MODEL.RECON_LOSS = CN()
_C.MODEL.RECON_LOSS.NAME = "l1"
_C.MODEL.RECON_LOSS.RENORMALIZE_DATA = True
_C.MODEL.RECON_LOSS.WEIGHT = 1.0

# -----------------------------------------------------------------------------
# Cascade Model
# -----------------------------------------------------------------------------
_C.MODEL.CASCADE = CN()
_C.MODEL.CASCADE.RECON_MODEL_NAME = ""
_C.MODEL.CASCADE.SEG_MODEL_NAME = ""
# Use magnitude image for semantic segmentation
_C.MODEL.CASCADE.USE_MAGNITUDE = False
# Zero-fill output reconstruction. Not needed if zero padding is done prior to recon.
_C.MODEL.CASCADE.ZERO_FILL = False
_C.MODEL.CASCADE.SEG_NORMALIZE = ""
# Iterative teacher forcing algorithm.
# PERIOD=0 indicates not to use ITFS
_C.MODEL.CASCADE.ITFS = CN()
_C.MODEL.CASCADE.ITFS.PERIOD = 0

_C.MODEL.SEG = CN()
_C.MODEL.SEG.CLASSES = ()
_C.MODEL.SEG.INCLUDE_BACKGROUND = False
_C.MODEL.SEG.ACTIVATION = "sigmoid"
_C.MODEL.SEG.LOSS_NAME = "DiceLoss"
_C.MODEL.SEG.LOSS_WEIGHT = 1.0
_C.MODEL.SEG.MODEL = CN()
_C.MODEL.SEG.MODEL.VNET_MONAI = _VNET_MONAI.clone()
_C.MODEL.SEG.MODEL.UNET_MONAI = _UNET_MONAI.clone()
_C.MODEL.SEG.MODEL.DYNUNET_MONAI = _DYNUNET_MONAI.clone()
_C.MODEL.SEG.USE_MAGNITUDE = True
_C.MODEL.SEG.IN_CHANNELS = None
# _C.MODEL.SEG.MODEL.NNUNET_MONAI = _NNUNET_MONAI.clone()

# Image normalization options.
# See skm_tea/data/transforms/transform.py for more details.
_C.MODEL.NORMALIZER = CN()
_C.MODEL.NORMALIZER.NAME = "TopMagnitudeNormalizer"
# Keywords to normalize
# See skm_tea/data/transforms/transform.py for more details
_C.MODEL.NORMALIZER.KEYWORDS = ()

# -----------------------------------------------------------------------------
# TB Recon Model
# -----------------------------------------------------------------------------
_C.MODEL.TB_RECON = _TB_RECON.clone()

# -----------------------------------------------------------------------------
# UNET model
# -----------------------------------------------------------------------------
_C.MODEL.UNET = CN()
_C.MODEL.UNET.OUT_CHANNELS = 2
_C.MODEL.UNET.IN_CHANNELS = 2
_C.MODEL.UNET.CHANNELS = 32
_C.MODEL.UNET.NUM_POOL_LAYERS = 4
_C.MODEL.UNET.DROPOUT = 0.0
_C.MODEL.UNET.NORMALIZE = False
# BLOCK_ORDER only applies to GeneralizedUNet
_C.MODEL.UNET.BLOCK_ORDER = ("conv", "relu", "conv", "relu", "batchnorm", "dropout")

# -----------------------------------------------------------------------------
# Denoising model
# -----------------------------------------------------------------------------
_C.MODEL.DENOISING = CN()
_C.MODEL.DENOISING.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.DENOISING.NOISE = CN()
# Noise standard deviation to use for augmentations.
_C.MODEL.DENOISING.NOISE.STD_DEV = (1,)
# When fully sampled kspace is available, if True, perform denoising on the
# fully sampled kspace. If False, denoising will be performed on the
# randomly generated undersampled kspace.
_C.MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET = True
# Same as above, but at eval time (e.g. validation).
# Defaults to MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET
_C.MODEL.DENOISING.NOISE.USE_FULLY_SAMPLED_TARGET_EVAL = None

# -----------------------------------------------------------------------------
# Compressed Sensing (CS) model
# -----------------------------------------------------------------------------
_C.MODEL.CS = CN()
_C.MODEL.CS.REGULARIZATION = 0.005
_C.MODEL.CS.MAX_ITER = 200

# -----------------------------------------------------------------------------
# Noise2Recon (N2R) model
# -----------------------------------------------------------------------------
_C.MODEL.N2R = CN()
_C.MODEL.N2R.META_ARCHITECTURE = "GeneralizedUnrolledCNN"
_C.MODEL.N2R.USE_SUPERVISED_CONSISTENCY = False
# Use MODEL.CONSISTENCY.AUG.NOISE.STD_DEV to control noise parameters.

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training. Must be registered in DatasetCatalog
_C.DATASETS.TRAIN = ()
# List of the dataset names for validation. Must be registered in DatasetCatalog
_C.DATASETS.VAL = ()
# List of the dataset names for testing. Must be registered in DatasetCatalog
_C.DATASETS.TEST = ()
# qDESS specific dataset args
_C.DATASETS.QDESS = CN()
_C.DATASETS.QDESS.ECHO_KIND = "echo1"
_C.DATASETS.QDESS.DATASET_TYPE = ""
_C.DATASETS.QDESS.KWARGS = ()


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# Number of batches to prefetch per worker
_C.DATALOADER.PREFETCH_FACTOR = 2
# If True, the dataloader will drop the last batch.
_C.DATALOADER.DROP_LAST = True
# Subsample training data to simulate data limited scenarios.
_C.DATALOADER.SUBSAMPLE_TRAIN = CN()
# Number of training examples to retain. All others will be force dropped
# meaning they will not be used for any training, even w/ semi-supervised N2R
# framework
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL = -1
# Number of training examples to retain based on metadata properties.
# e.g. (("acquisition", {"AXT2": 15, "AXT1": 10}),) will only keep
# 15 AXT2 scans and 10 AXT1 scans where AXT2 and AXT1 are categories in the
# "acquisition" metadata field.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL_BY_GROUP = ()
# Number of scans out of total to undersample. If NUM_TOTAL is not -1, must be
# less than NUM_TOTAL.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED = 0
# Number of scans with ground truth for validation.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL = -1
# Number of validation examples to retain based on metadata properties.
_C.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL_BY_GROUP = ()
# Seed for shuffling data. Should always be deterministic
_C.DATALOADER.SUBSAMPLE_TRAIN.SEED = 1000
# Options: "" (defaults to random sampling), "AlternatingSampler"
_C.DATALOADER.SAMPLER_TRAIN = ""
# AlternatingSampler config parameters.
_C.DATALOADER.ALT_SAMPLER = CN()
_C.DATALOADER.ALT_SAMPLER.PERIOD_SUPERVISED = 1
_C.DATALOADER.ALT_SAMPLER.PERIOD_UNSUPERVISED = 1
# GroupSampler config parameters.
_C.DATALOADER.GROUP_SAMPLER = CN()
_C.DATALOADER.GROUP_SAMPLER.BATCH_BY = ()
_C.DATALOADER.GROUP_SAMPLER.AS_BATCH_SAMPLER = False
# Paired tuple of data keys and H5DF keys. Empty tuple will result in default keys being used.
# e.g. (("target", "espirit_recon"), ("maps", "espirit_maps"))
# See data/slice_data.py for more information.
_C.DATALOADER.DATA_KEYS = ()

_C.DATALOADER.FILTER = CN()
# Paired tuple of key and values for filtering. Multiple values should be specified as tuple.
# e.g. (("num_slices", 30),) will only keep data with number of slices == 30
# Field must appear in all dataset dicts.
# Training, validation, and testing data is filtered by this field.
_C.DATALOADER.FILTER.BY = ()


# -----------------------------------------------------------------------------
# Augmentations/Transforms
# -----------------------------------------------------------------------------
_C.AUG_TRAIN = CN()
_C.AUG_TRAIN.UNDERSAMPLE = CN()
_C.AUG_TRAIN.UNDERSAMPLE.NAME = "PoissonDiskMaskFunc"
_C.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS = (6,)
_C.AUG_TRAIN.UNDERSAMPLE.CALIBRATION_SIZE = 24
_C.AUG_TRAIN.UNDERSAMPLE.CENTER_FRACTIONS = ()
_C.AUG_TRAIN.UNDERSAMPLE.MAX_ATTEMPTS = 5
_C.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE = CN()
# Training masks to precompute.
_C.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.NUM = -1
_C.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.SEED = -1
_C.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.USE_MULTIPROCESSING = False
# Augment with noise. See MODEL.CONSISTENCY.AUG.NOISE for parameters.
_C.AUG_TRAIN.USE_NOISE = False
_C.AUG_TRAIN.NOISE_P = 0.2

_C.AUG_TEST = CN()
_C.AUG_TEST.UNDERSAMPLE = CN()
_C.AUG_TEST.UNDERSAMPLE.ACCELERATIONS = (6,)

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

_C.SOLVER.MAX_ITER = 20

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.BASE_LR = 1e-4
_C.SOLVER.MIN_LR = None
# See meddlr.solver.build for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = ""
_C.SOLVER.GAMMA = 1.0
# The iteration number to decrease learning rate by GAMMA.
_C.SOLVER.STEPS = ()

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

# Number of iterations over which to accumulate gradients. Must be > 0.
_C.SOLVER.GRAD_ACCUM_ITERS = 1

# Save a checkpoint after every this number of iterations
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.CHECKPOINT_MONITOR = "val_loss"

# Early stopping
# Patience is only supported in number of epochs.
_C.SOLVER.EARLY_STOPPING = CN()
_C.SOLVER.EARLY_STOPPING.MONITOR = "val_loss"
_C.SOLVER.EARLY_STOPPING.PATIENCE = 0
_C.SOLVER.EARLY_STOPPING.MIN_DELTA = 0.0

# Number of images per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.SOLVER.TRAIN_BATCH_SIZE = 16
_C.SOLVER.TEST_BATCH_SIZE = 16

# Bias functionality is identical to typical weights.
# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# The period over which to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 1
_C.TEST.QDESS_EVALUATOR = CN()
_C.TEST.QDESS_EVALUATOR.ADDITIONAL_PATHS = ()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []

# VALIDATION BEHAVES LIKE TEST-TIME. To make this configurable, uncomment this line.
# Validate with test-like functionality.
# If True, undersampling masks for every validation scan will be fixed
# given an acceleration.
# _C.TEST.VAL_AS_TEST = False

# Validation metrics.
_C.TEST.VAL_METRICS = CN()
_C.TEST.VAL_METRICS.RECON = ()
_C.TEST.VAL_METRICS.SEM_SEG = ()
# The period over which to flush evaluation results.
# Specify if you are getting OOM errors during evaluation.
# Set to 0 to disable.
_C.TEST.FLUSH_PERIOD = 0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = ""
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = -1
# The period (in terms of steps) for minibatch visualization at train time.
# Set to 0 to disable. Currently not functional.
# This will always be in number of iterations.
_C.VIS_PERIOD = 0
# The scale when referring to time generally in the config.
# Note that there are certain fields, which are explicitly at the iteration
# scale (e.g. GRAD_ACCUM_ITERS).
# Either "epoch" or "iter"
_C.TIME_SCALE = "epoch"
_C.CUDNN_BENCHMARK = False

# ---------------------------------------------------------------------------- #
# Config Description
# ---------------------------------------------------------------------------- #
_C.DESCRIPTION = CN()
# Brief description about config
_C.DESCRIPTION.BRIEF = ""
# The entity (team/individual) account for logging to Weights & Biases
_C.DESCRIPTION.ENTITY_NAME = ""
# Project name for logging to Weights & Biases
_C.DESCRIPTION.PROJECT_NAME = ""
# Experiment name for logging to Weights & Biases
_C.DESCRIPTION.EXP_NAME = ""
# Tags associated with experiment.
# e.g. "fastmri_knee_mc" for fastMRI dataset; "unrolled" for using unrolled network; etc.
_C.DESCRIPTION.TAGS = ()
