import logging
import os

import pytorch_lightning as pl
from meddlr.config.config import CfgNode
from meddlr.engine.trainer import convert_cfg_time_to_iter as _convert_cfg_time_to_iter
from meddlr.engine.trainer import format_as_iter
from meddlr.utils import env
from meddlr.utils.env import supports_wandb
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.utilities.distributed import rank_zero_only

from skm_tea.callbacks import PLPeriodicCheckpointer
from skm_tea.utils.pl_utils import LoggerCollection, TensorBoardLogger, WandbLogger

__all__ = ["PLDefaultTrainer"]


def convert_cfg_time_to_iter(cfg: CfgNode, iters_per_epoch: int):
    """Convert all config time-related parameters to iterations.

    Note:
        When adding to this list, be careful not to convert config parameters
        multiple times.
    """
    time_scale = cfg.TIME_SCALE

    cfg = _convert_cfg_time_to_iter(cfg.clone(), iters_per_epoch, ignore_missing=True).defrost()
    cfg.SOLVER.EARLY_STOPPING.PATIENCE = format_as_iter(
        cfg.SOLVER.EARLY_STOPPING.PATIENCE, iters_per_epoch, time_scale
    )

    cfg.TIME_SCALE = "iter"
    cfg.freeze()
    return cfg


class PLDefaultTrainer(pl.Trainer):
    def __init__(
        self,
        cfg,
        iters_per_epoch: int,
        log_gpu_memory=None,
        replace_sampler_ddp=False,
        num_gpus=0,
        resume=False,
        eval_only=False,
        **kwargs,
    ):
        logger = logging.getLogger("skm_tea")

        self.eval_only = eval_only
        if "limit_train_batches" in kwargs:
            iters_per_epoch = kwargs["limit_train_batches"]
        cfg = convert_cfg_time_to_iter(cfg, iters_per_epoch)
        self.cfg = cfg

        callbacks = self.build_callbacks()  # includes user-specified callbacks
        kwargs["callbacks"] = callbacks

        if resume:
            assert not kwargs.get(
                "resume_from_checkpoint", None
            ), "Cannot specify resume=True and resume_from_checkpoint"
            resume_from_checkpoint = self.configure_resume(callbacks)
            logger.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            kwargs["resume_from_checkpoint"] = resume_from_checkpoint

        early_stopping_callback = self.build_early_stopping(iters_per_epoch)
        if early_stopping_callback:
            callbacks.append(early_stopping_callback)

        # Hacky way to get around the definition of "step" as optimizer.step in pt-lightning.
        # Without this the training time would be scaled by a factor of SOLVER.GRAD_ACCUM_ITERS.
        max_steps = cfg.SOLVER.MAX_ITER // cfg.SOLVER.GRAD_ACCUM_ITERS

        # Default arguments based on Trainer. Any keyword args provided will overwrite these.
        args = dict(
            logger=self.build_logger() if not self.eval_only else False,
            default_root_dir=cfg.OUTPUT_DIR,
            max_steps=max_steps,
            # TODO Issue #4406: https://github.com/PyTorchLightning/pytorch-lightning/issues/4406
            val_check_interval=min(
                cfg.TEST.EVAL_PERIOD, kwargs.get("limit_train_batches", float("inf"))
            ),
            accumulate_grad_batches=cfg.SOLVER.GRAD_ACCUM_ITERS,
            log_gpu_memory=log_gpu_memory,
            checkpoint_callback=False,
            sync_batchnorm=False,
            profiler=SimpleProfiler(dirpath=cfg.OUTPUT_DIR, filename="profile.txt"),
            log_every_n_steps=5,
            replace_sampler_ddp=replace_sampler_ddp,
            deterministic=env.is_repro(),
        )
        if num_gpus > 0:
            args.update({"gpus": num_gpus, "auto_select_gpus": True})

        args.update(kwargs)
        super().__init__(**args)

    def build_early_stopping(self, iters_per_epoch):
        monitor = self.cfg.SOLVER.EARLY_STOPPING.MONITOR
        patience = self.cfg.SOLVER.EARLY_STOPPING.PATIENCE
        min_delta = self.cfg.SOLVER.EARLY_STOPPING.MIN_DELTA

        if patience == 0:
            return False

        patience = patience / iters_per_epoch
        assert (
            self.cfg.TIME_SCALE == "iter" and patience > 0 and int(patience) == patience
        ), f"Got time scale '{self.cfg.TIME_SCALE}' and patience '{patience}'"
        return EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=True)

    @rank_zero_only
    def build_logger(self):
        cfg = self.cfg
        version = ""
        loggers = [
            CSVLogger(cfg.OUTPUT_DIR, name="", version=version),
            TensorBoardLogger(cfg.OUTPUT_DIR, name="", version=version, log_graph=False),
        ]
        if supports_wandb():
            import wandb

            loggers.append(WandbLogger(experiment=wandb.run))

        return LoggerCollection(loggers)

    def build_callbacks(self, **kwargs):
        """Append default callbacks to list of user-defined callbacks."""
        cfg = self.cfg
        callbacks = list(kwargs.get("callbacks", []))
        if "checkpoint_callback" not in kwargs and not any(
            isinstance(x, PLPeriodicCheckpointer) for x in callbacks
        ):
            callbacks.append(
                PLPeriodicCheckpointer(
                    frequency=cfg.SOLVER.CHECKPOINT_PERIOD,
                    filepath=os.path.join(cfg.OUTPUT_DIR, "{global_step:07d}-{epoch:03d}"),
                    save_after_val=True,
                )
            )

        return callbacks

    def configure_resume(self, callbacks):
        """Configure setup for resume.

        Currently finds the latest epoch and resumes from there.
        """
        # cfg = self.cfg
        checkpointer = [x for x in callbacks if isinstance(x, PLPeriodicCheckpointer)]
        if len(checkpointer) == 0:
            raise ValueError("Resuming training only works with PLPeriodicCheckpointer")
        elif len(checkpointer) > 1 and any(
            cp.dirpath != checkpointer[0].dirpath for cp in checkpointer
        ):
            raise ValueError("Found more than one checkpointer with different save directories")
        return checkpointer[0].get_latest()
