"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find
them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in
their projects.
"""
import argparse
import logging
import os
import re
from typing import Sequence

import torch
from meddlr.config.config import CfgNode
from meddlr.engine.defaults import init_reproducible_mode as _init_reproducible_mode
from meddlr.utils import comm
from meddlr.utils.collect_env import collect_env_info
from meddlr.utils.env import get_available_gpus, seed_all_rng
from meddlr.utils.logger import setup_logger

from skm_tea.utils import env
from skm_tea.utils.general import format_exp_version

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no-cover
    wandb = None
    Run = None
    _WANDB_AVAILABLE = False


__all__ = ["default_argument_parser", "default_setup"]

_path_mgr = env.get_path_manager()


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="SKM-TEA Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument(
        "--restart-iter",
        action="store_true",
        help="restart iteration count when loading checkpointed weights",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus. overrided by --devices"
    )
    parser.add_argument("--devices", type=int, nargs="*", default=None)
    parser.add_argument("--debug", action="store_true", help="use debug mode")
    parser.add_argument("--sharded", action="store_true", help="use model sharding")
    parser.add_argument(
        "--reproducible", "--repro", action="store_true", help="activate reproducible mode"
    )
    parser.add_argument("--eval-on-cpu", action="store_true", help="evaluate metrics on cpu")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(
    cfg: CfgNode, args: argparse.Namespace, save_cfg: bool = True, use_lightning=False
):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the meddlr and skm-tea loggers.
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    4. Enables debug mode if ``args.debug==True``
    5. Enables reproducible model if ``args.reproducible==True``

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
        save_cfg (bool, optional): If `True`, writes config to `cfg.OUTPUT_DIR`.

    Note:
        Project-specific environment variables are modified by this function.
        ``cfg`` is also modified in-place.
    """
    rank = comm.get_rank()
    is_repro_mode = (
        env.is_repro() if env.is_repro() else (hasattr(args, "reproducible") and args.reproducible)
    )
    eval_only = hasattr(args, "eval_only") and args.eval_only

    # Update config parameters before saving.
    cfg.defrost()
    cfg.format_fields()
    output_dir = _path_mgr.get_local_path(cfg.OUTPUT_DIR)
    make_new_version = not (
        (hasattr(args, "eval_only") and args.eval_only)
        or args.resume
        or rank != 0
        or re.match("version_[0-9]*$", output_dir)
    )
    if not make_new_version and not os.path.isdir(output_dir):
        raise ValueError(
            f"Tried to resume or evaluate on empty directory. " f"{output_dir} does not exist."
        )
    if args.debug:
        output_dir = (
            os.path.join(output_dir, "debug")
            if os.path.basename(output_dir).lower() != "debug"
            else output_dir
        )
    else:
        output_dir = format_exp_version(output_dir, new_version=make_new_version)
    cfg.OUTPUT_DIR = output_dir

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        _path_mgr.mkdirs(output_dir)

    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    setup_logger(output_dir, distributed_rank=rank)
    logger = setup_logger(output_dir, distributed_rank=rank, name="skm_tea", abbrev_name="st")

    if is_repro_mode:
        init_reproducible_mode(cfg, eval_only)

    cfg.freeze()

    if args.debug:
        os.environ["MEDDLR_DEBUG"] = "True"
        logger.info("Running in debug mode")
    if use_lightning:
        os.environ["MEDDLR_PT_LIGHTNING"] = "True"
        logger.info("Running with pytorch lightning")
    if is_repro_mode:
        os.environ["MEDDLR_REPRO"] = "True"
        logger.info("Running in reproducible mode")

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, _path_mgr.open(args.config_file, "r").read()
            )
        )

    if not use_lightning:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            if args.devices:
                gpus = args.devices
                if not isinstance(gpus, Sequence):
                    gpus = [gpus]
            else:
                gpus = get_available_gpus(args.num_gpus)

        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in gpus])

        # TODO: Remove this and find a better way to launch the script
        # with the desired gpus.
        if gpus[0] >= 0:
            torch.cuda.set_device(gpus[0])

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir and save_cfg:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with _path_mgr.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of
    # typical validation set.
    if not eval_only:
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    # Initialize Weights and Biases run.
    # Resume if we are evaluating or resuming
    # if use_wandb:
    #     _init_wandb_experiment(cfg, resume=not make_new_version)


def init_reproducible_mode(cfg: CfgNode, eval_only: bool = False):
    logger = logging.getLogger(__name__)

    # Set all seeds in the config if they are not set.
    # NOTE: The seed for precomputing masks is not set due to
    # existing issues with numba stalling with fixed seeds at large.
    # N values.
    precompute_seed = cfg.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.SEED

    _init_reproducible_mode(cfg, eval_only=eval_only)

    if precompute_seed == -1:
        logger.warning(
            "Seed for precomputing masks (AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.SEED) "
            "could not be set. This may result in non-reproducible behavior."
        )
        cfg.defrost()
        cfg.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.SEED = precompute_seed
        cfg.freeze()
