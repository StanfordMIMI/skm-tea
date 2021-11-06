import os
import re

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem

from skm_tea.utils import env

__all__ = ["PLPeriodicCheckpointer"]

_PATH_MANAGER = env.get_path_manager()


class PLPeriodicCheckpointer(pl.Callback):
    LAST_CHECKPOINT_FILE = "last_checkpoint"

    def __init__(self, frequency, filepath, prefix: str = "", save_after_val=False):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.frequency = frequency
        self.prefix = prefix
        self.save_after_val = save_after_val

        if filepath:
            self._fs = get_filesystem(filepath)
        else:
            self._fs = get_filesystem("")  # will give local fileystem

        if self._fs.isdir(filepath):
            self.dirpath, self.filename = filepath, "{epoch}"
        else:
            if self._fs.protocol == "file":  # dont normalize remote paths
                filepath = os.path.realpath(filepath)
            self.dirpath, self.filename = os.path.split(filepath)
        self._fs.makedirs(self.dirpath, exist_ok=True)

    def format_checkpoint_name(self, global_step, epoch, metrics):
        # check if user passed in keys to the string
        groups = re.findall(r"(\{.*?)[:\}]", self.filename)

        if len(groups) == 0:
            # default name
            filename = f"{self.prefix}_ckpt_epoch_{epoch}"
        else:
            metrics["epoch"] = epoch
            metrics["global_step"] = global_step
            filename = self.filename
            for tmp in groups:
                name = tmp[1:]
                filename = filename.replace(tmp, name + "={" + name)
                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)
        filepath = os.path.join(self.dirpath, self.prefix + filename + ".ckpt")
        return filepath

    @rank_zero_only
    def checkpoint(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        # metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        ckpt_name_metrics = trainer.logged_metrics

        filepath = self.format_checkpoint_name(global_step, epoch, ckpt_name_metrics)
        trainer.save_checkpoint(filepath)
        self.tag_last_checkpoint(os.path.basename(filepath))

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.dirpath, "last_checkpoint")
        return _PATH_MANAGER.exists(save_file)

    def get_latest(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.dirpath, "last_checkpoint")
        try:
            with _PATH_MANAGER.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.dirpath, last_saved)  # pyre-ignore

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.
        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        save_file = os.path.join(self.dirpath, "last_checkpoint")
        with _PATH_MANAGER.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore

    @rank_zero_only
    def on_batch_end(self, trainer: pl.Trainer, pl_module):
        """Check if we should save a checkpoint after every train batch"""
        if (trainer.global_step + 1) % self.frequency == 0:
            self.checkpoint(trainer, pl_module)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if self.save_after_val:
            self.checkpoint(trainer, pl_module)
