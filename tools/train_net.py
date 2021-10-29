"""SKM-TEA training and evaluation workflow.

This is the sample script for training and evaluating methods on
the SKM-TEA dataset. This script is compatible with all config-driven
functionality in the skm_tea package.

This script can be useful for scalable training and evaluation.
For a more interactive experience with the dataset, see the
notebooks included in the repository.
"""

import os
import warnings
from typing import Sequence

import torch
from ss_recon.config import get_cfg
from ss_recon.evaluation.testing import check_consistency, find_weights
from ss_recon.utils import comm
from ss_recon.utils.env import supports_wandb

from skm_tea import config  # noqa: F401 (required for setting default config)
from skm_tea.engine import (
    PLDefaultTrainer,
    default_argument_parser,
    default_setup,
    qDESSModule,
    qDESSSemSegModel,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    pass


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if opts and opts[0] == "--":
        opts = opts[1:]
    cfg.merge_from_list(opts)
    cfg.freeze()

    if not cfg.OUTPUT_DIR:
        raise ValueError("OUTPUT_DIR not specified")

    default_setup(cfg, args, use_lightning=True, save_cfg=not args.eval_only)

    # TODO: Add suppport for resume.
    if comm.is_main_process() and supports_wandb() and not args.eval_only:
        exp_name = cfg.DESCRIPTION.EXP_NAME
        if not exp_name:
            warnings.warn("DESCRIPTION.EXP_NAME not specified. Defaulting to basename...")
            exp_name = os.path.basename(cfg.OUTPUT_DIR)
        wandb.init(
            project=cfg.DESCRIPTION.PROJECT_NAME,
            name=exp_name,
            config=cfg,
            sync_tensorboard=True,
            job_type="training",
            dir=cfg.OUTPUT_DIR,
            entity=cfg.DESCRIPTION.ENTITY_NAME,
            tags=cfg.DESCRIPTION.TAGS,
            notes=cfg.DESCRIPTION.BRIEF,
        )
    return cfg


def main(args, pl_module=None):
    cfg = setup(args)

    if args.devices:
        gpus = args.devices
        if not isinstance(gpus, Sequence):
            gpus = [gpus]
        auto_select_gpus = False
        num_gpus = len(gpus)
    else:
        gpus = num_gpus = args.num_gpus if args.num_gpus >= 1 else None
        auto_select_gpus = True

    distributed_backend = "ddp" if num_gpus > 1 else None
    if pl_module is None:
        pl_module = qDESSModule if "recon" in cfg.MODEL.TASKS else qDESSSemSegModel
    model = pl_module(cfg, num_parallel=num_gpus, eval_on_cpu=args.eval_on_cpu)
    trainer = PLDefaultTrainer(
        cfg,
        model.iters_per_epoch,
        gpus=gpus,
        auto_select_gpus=auto_select_gpus,
        resume=args.resume,
        distributed_backend=distributed_backend,
        eval_only=args.eval_only,
    )
    if not args.eval_only:
        trainer.fit(model)

    # Evaluation
    weights, _, _ = find_weights(
        cfg, criterion="loss", iter_limit=None, file_name_fmt="global_step={:07d}-.*\.ckpt"
    )
    state_dict = torch.load(weights)["state_dict"]
    model.to(cfg.MODEL.DEVICE)
    model.load_state_dict(state_dict)
    check_consistency(state_dict, model)
    trainer.test(model=model)[0]


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)