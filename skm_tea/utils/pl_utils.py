"""Pytorch lightning utils."""
import logging
import os
from typing import Dict, Optional

import numpy as np
from meddlr.evaluation.testing import flatten_results_dict
from pytorch_lightning.loggers import LoggerCollection as _LoggerCollection
from pytorch_lightning.loggers import TensorBoardLogger as _TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger as _WanbLogger
from pytorch_lightning.utilities import rank_zero_only

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no-cover
    wandb = None
    _WANDB_AVAILABLE = False

_logger = logging.getLogger(__name__)


class TensorBoardLogger(_TensorBoardLogger):
    @rank_zero_only
    def log_images(
        self, images: Dict[str, np.ndarray], step: Optional[int] = None, data_format: str = "CHW"
    ) -> None:
        """
        Args:
            images (Dict[str, ndarray]): Pairs of image names and images.
                Each image must be an `uint8` or `float`.
            step (int, optional): Step number at which the images should be recorded.
            data_format (str, optional): Either "CHW", "HW", or "HWC"
        """
        assert data_format in ("CHW", "HW", "HWC")
        for img_name, img in images.items():
            self.experiment.add_image(img_name, img, step, dataformats=data_format)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        from meddlr.evaluation.testing import flatten_results_dict

        metrics = flatten_results_dict(metrics)
        super().log_metrics(metrics=metrics, step=step)


class WandbLogger(_WanbLogger):
    """An extension of the W&B Logger for Pytorch lightning.

    This logger supports logging images. It also prevents logging metrics and
    images when sync_tensorboard is enabled.

    Logging metrics using this logger and initializing the W&B run with `sync_tensorboard=True`
    results in double logging of metrics. This is not ideal.
    """

    def __init__(
        self,
        *args,
        sync_tensorboard: Optional[bool] = None,
        ignore_sync_tensorboard=False,
        **kwargs
    ):
        if sync_tensorboard is None:
            sync_tensorboard = os.environ.get("WANDB_SYNC_TENSORBOARD", "false").lower() == "true"
        self.sync_tensorboard = sync_tensorboard
        if ignore_sync_tensorboard and sync_tensorboard:
            _logger.warn("Ignoring sync tensorboard may double log metrics")

        # Logs images if not synced with tensorboard or ignoring syncing
        # and writing anyway (not suggested).
        self._do_log: bool = not sync_tensorboard or ignore_sync_tensorboard
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not self._do_log:
            return

        metrics = flatten_results_dict(metrics)
        return super().log_metrics(metrics, step)

    @rank_zero_only
    def log_images(
        self,
        images: Dict[str, np.ndarray],
        step: Optional[int] = None,
        data_format: str = "CHW",
        commit=False,
    ) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        # Configure this more intelligently to track what images are actually
        # being written to tensorboard. Then skip all images that are already logged.
        if not self._do_log:
            return

        assert data_format in ("CHW", "HW", "HWC")
        imgs = {
            img_name: wandb.Image(self._to_hwc(img, data_format))
            for img_name, img in images.items()
        }

        self.experiment.log(
            {"global_step": step, **imgs} if step is not None else imgs, commit=commit
        )

    def _to_hwc(self, _img: np.ndarray, _data_format):
        assert _data_format in ("CHW", "HW", "HWC")

        if _data_format == "HWC":
            return _img
        elif _data_format == "HW":
            return _img[..., np.newaxis]
        elif _data_format == "CHW":
            return np.transpose(_img, (1, 2, 0))

    def make_image(self, img: np.ndarray, data_format: str = "CHW", **kwargs):
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        return wandb.Image(self._to_hwc(img, data_format), **kwargs)


class LoggerCollection(_LoggerCollection):
    """Default logger collection for this project.

    Supports default logging of images as well.
    """

    def log_images(
        self, images: Dict[str, np.ndarray], step: Optional[int] = None, data_format: str = "CHW"
    ) -> None:
        for logger in self._logger_iterable:
            if hasattr(logger, "log_images"):
                logger.log_images(images, step, data_format)
