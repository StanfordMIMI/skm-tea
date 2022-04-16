import logging
import os

import meddlr.ops as oF
import numpy as np
import torch
from meddlr.config.config import CfgNode
from meddlr.data.transforms.transform import normalize_affine, unnormalize_affine
from meddlr.evaluation.evaluator import DatasetEvaluators
from meddlr.evaluation.recon_evaluation import ReconEvaluator
from meddlr.evaluation.seg_evaluation import SemSegEvaluator
from meddlr.modeling.loss_computer import build_loss_computer
from meddlr.ops import complex as cplx
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from skm_tea.data.data_module import SkmTeaDataModule
from skm_tea.engine.modules.recon import ReconModule
from skm_tea.evaluation.qdess_evaluation import SkmTeaEvaluator
from skm_tea.losses import SegLossComputer
from skm_tea.modeling.build import build_model

logger = logging.getLogger(__name__)

__all__ = ["SkmTeaModule", "SkmTeaSemSegModule"]


class SkmTeaModule(ReconModule):
    def __init__(
        self,
        cfg: CfgNode,
        num_parallel=1,
        eval_on_cpu: bool = False,
        deployment: bool = False,
        **kwargs,
    ):
        self.tasks = cfg.MODEL.TASKS
        super().__init__(
            cfg, num_parallel, eval_on_cpu=eval_on_cpu, deployment=deployment, **kwargs
        )
        self.std_log = logging.getLogger(__name__)
        if not deployment:
            self._datamodule_ = self._build_datamodule(self.cfg)
            self._datamodule_.setup()
        self.seg_classes = self.cfg.MODEL.SEG.CLASSES

    def _build_datamodule(self, cfg):
        return SkmTeaDataModule(cfg, self.tasks, self.distributed)

    def train_dataloader(self, cfg=None):
        if not cfg:
            cfg = self.cfg
        if not hasattr(self, "_datamodule_") or not self._datamodule_:
            datamodule = self._build_datamodule(cfg)
            datamodule.prepare_data()
            datamodule.setup()
        else:
            datamodule = self._datamodule_

        return datamodule.train_dataloader(cfg, self.distributed)

    def val_dataloader(self):
        return self._datamodule_.val_dataloader(self.distributed)

    def test_dataloader(self):
        return self._datamodule_.test_dataloader(self.distributed)

    def build_loss_computer(self, cfg):
        assert self.tasks == ("recon",)
        return build_loss_computer(cfg, "BasicLossComputer")

    @torch.no_grad()
    @rank_zero_only
    def visualize(self, inputs, outputs, prefix="train"):
        channels = (
            ("echo1", "echo2") if self.cfg.DATASETS.QDESS.ECHO_KIND == "echo1-echo2-mc" else None
        )
        # Logs recon images.
        super().visualize(inputs=inputs, outputs=outputs, prefix=prefix, channels=channels)

        wandb_logger = self.wandb_logger()

        # Log segmentations/detection images (if applicable)
        if "sem_seg" in self.tasks and wandb_logger is not None:
            class_labels = {i + 1: cat for i, cat in enumerate(self.seg_classes)}
            class_labels[0] = "background"
            gt_seg = oF.one_hot_to_categorical(
                inputs["sem_seg"][0, ...].cpu().numpy(), channel_dim=0
            )
            pred_seg = oF.pred_to_categorical(
                outputs["sem_seg_logits"][0, ...].cpu().numpy(),
                activation=self.cfg.MODEL.SEG.ACTIVATION,
                threshold=0.5,
                channel_dim=0,
            )

            masks = {
                "predictions": {"mask_data": pred_seg, "class_labels": class_labels},
                "ground_truth": {"mask_data": gt_seg, "class_labels": class_labels},
            }

            visualizations = {}

            # Make overlay on predicted recon.
            pred = outputs["pred"][0].squeeze()
            if cplx.is_complex(pred) or cplx.is_complex_as_real(pred):
                pred = cplx.abs(pred)
            pred = pred.cpu().numpy()
            pred -= np.min(pred)
            pred /= np.max(pred)

            visualizations["seg_on_recon"] = wandb_logger.make_image(
                pred, data_format="HW", masks=masks
            )

            # Make overlay on target.
            target = outputs["target"][0].squeeze()
            if cplx.is_complex(target) or cplx.is_complex_as_real(target):
                target = cplx.abs(target)
            target = target.cpu().numpy()
            target -= np.min(target)
            target /= np.max(target)
            visualizations["seg_on_tgt"] = wandb_logger.make_image(
                target, data_format="HW", masks=masks
            )

            visualizations = {f"{prefix}/{name}": img for name, img in visualizations.items()}
            wandb_logger.experiment.log({"global_step": self.trainer.global_step, **visualizations})

    @rank_zero_only
    def wandb_logger(self):
        if isinstance(self.logger, LoggerCollection):
            loggers = list(self.logger._logger_iterable)
            for lgr in loggers:
                if isinstance(lgr, WandbLogger):
                    return lgr
            return None
        elif isinstance(self.logger, WandbLogger):
            return self.logger
        else:
            return None

    def build_evaluator(
        self, cfg, dataset_name, group_by_scan=False, save_data=False, **kwargs
    ) -> DatasetEvaluators:
        evaluators = []

        # TODO: More efficient to compute with each separately
        aggregate_scans = not self.trainer.sanity_checking and not self.trainer.fast_dev_run
        output_dir = os.path.join(cfg.OUTPUT_DIR, "test_results", dataset_name)
        skip_rescale = False
        distributed = self.distributed

        recon_metrics = kwargs.pop("recon_metrics", cfg.TEST.VAL_METRICS.RECON)
        seg_metrics = kwargs.pop("sem_seg_metrics", cfg.TEST.VAL_METRICS.SEM_SEG)
        if kwargs.pop("metrics", "") is None:
            recon_metrics = None
            seg_metrics = None
        flush_period = kwargs.pop("flush_period", cfg.TEST.FLUSH_PERIOD)
        to_cpu = kwargs.pop("to_cpu", self.eval_on_cpu)

        echo_kind = cfg.DATASETS.QDESS.ECHO_KIND
        structure_channel_by = "echo" if echo_kind == "echo1+echo2" else None

        # Reconstruction tasks should only perform qMRI evaluation if both
        # echos are present or if path to second echo scans are specified
        if "recon" in self.tasks:
            use_qmri = aggregate_scans and echo_kind != "echo2"
            if (
                use_qmri
                and echo_kind == "echo1"
                and "recon/echo2" not in cfg.TEST.QDESS_EVALUATOR.ADDITIONAL_PATHS
            ):
                use_qmri = False
        else:
            use_qmri = aggregate_scans

        if self._stage == "test":
            evaluators.append(
                SkmTeaEvaluator(
                    dataset_name,
                    cfg,
                    distributed=distributed,
                    sync_outputs=False,
                    aggregate_scans=aggregate_scans,
                    output_dir=output_dir,
                    save_scans=save_data,
                    recon_metrics=recon_metrics,
                    sem_seg_metrics=seg_metrics,
                    flush_period=flush_period,
                    to_cpu=False,
                    tasks=self.tasks,
                    use_qmri=use_qmri,
                )
            )
        else:
            if "recon" in self.tasks:
                evaluators.append(
                    ReconEvaluator(
                        dataset_name,
                        cfg,
                        distributed=distributed,
                        sync_outputs=False,
                        aggregate_scans=aggregate_scans,
                        output_dir=output_dir,
                        skip_rescale=skip_rescale,
                        group_by_scan=group_by_scan,
                        save_scans=save_data,
                        metrics=recon_metrics,
                        flush_period=flush_period,
                        to_cpu=to_cpu,
                        eval_in_process=True,
                        structure_channel_by=structure_channel_by,
                    )
                )

            if "sem_seg" in self.tasks:
                evaluators.append(
                    SemSegEvaluator(
                        dataset_name,
                        cfg,
                        distributed=distributed,
                        sync_outputs=False,
                        output_dir=output_dir,
                        aggregate_scans=True,
                        group_by_scan=group_by_scan,
                        save_seg=save_data,
                        metrics=seg_metrics,
                        flush_period=flush_period,
                        to_cpu=to_cpu,
                    )
                )

        return DatasetEvaluators(evaluators)


class SkmTeaSemSegModule(SkmTeaModule):
    _ALIASES = ["qDESSSemSegModel"]

    def __init__(self, cfg, num_parallel=1, eval_on_cpu: bool = False, **kwargs):
        if cfg.MODEL.TASKS != ("sem_seg",):
            raise ValueError(f"{type(self).__name__} only supports semantic segmentation.")
        super().__init__(cfg, num_parallel, eval_on_cpu=eval_on_cpu, **kwargs)

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        use_magnitude = cfg.MODEL.SEG.USE_MAGNITUDE
        num_classes = len(cfg.MODEL.SEG.CLASSES)
        if num_classes == 0:
            raise ValueError("No classes found for segmentation")
        in_channels = cfg.MODEL.SEG.IN_CHANNELS
        if in_channels is None:
            in_channels = 1 if use_magnitude else 2

        model = build_model(cfg, in_channels=in_channels, out_channels=num_classes)
        return model

    def forward(self, inputs):
        if "target" in inputs:
            image = inputs["target"]  # segment off the target image.
            image = image.squeeze(-1)

            # Take magnitude
            if self.cfg.MODEL.SEG.USE_MAGNITUDE:
                image = cplx.abs(image)  # N x Y x X
                shape = (image.shape[0], 1) + image.shape[1:]
                image = image.view(shape).contiguous()  # N x 1 x Y x X

                # Normalize by volume.
                shape = (image.shape[0],) + (1,) * (image.ndim - 1)
                mean, std = inputs["mean"].view(shape), inputs["std"].view(shape)
                vol_mean, vol_std = (
                    inputs["stats"]["target"]["vol_mean"].view(shape),
                    inputs["stats"]["target"]["vol_std"].view(shape),
                )
                image = normalize_affine(unnormalize_affine(image, mean, std), vol_mean, vol_std)
                inputs["mean"], inputs["std"] = vol_mean, vol_std
            else:
                order = (0,) + (image.ndim - 1,) + tuple(range(1, image.ndim - 1))
                image = image.permute(order).contiguous()  # N x 2 x Y x X
        else:
            image = inputs["image"]

        outputs = self.model(image)
        if not isinstance(outputs, dict):
            outputs = {"sem_seg_logits": outputs}

        return outputs

    def build_loss_computer(self, cfg):
        return SegLossComputer(cfg)

    @torch.no_grad()
    @rank_zero_only
    def visualize(self, inputs, outputs, prefix="train"):
        wandb_logger = self.wandb_logger()

        # Log segmentations/detection images (if applicable)
        if wandb_logger is not None:
            class_labels = {i + 1: cat for i, cat in enumerate(self.seg_classes)}
            class_labels[0] = "background"
            gt_seg = oF.one_hot_to_categorical(
                inputs["sem_seg"][0, ...].cpu().numpy(), channel_dim=0
            )
            pred_seg = oF.pred_to_categorical(
                outputs["sem_seg_logits"][0, ...].cpu().numpy(),
                activation=self.cfg.MODEL.SEG.ACTIVATION,
                threshold=0.5,
                channel_dim=0,
            )

            masks = {
                "predictions": {"mask_data": pred_seg, "class_labels": class_labels},
                "ground_truth": {"mask_data": gt_seg, "class_labels": class_labels},
            }

            visualizations = {}
            # Make overlay on target.
            if "target" in inputs:
                target = inputs["target"][0].squeeze()
            else:
                target = inputs["image"][0].squeeze()
            if cplx.is_complex(target) or cplx.is_complex_as_real(target):
                target = cplx.abs(target)

            # TODO: Fix this to actually check if channel dimension still exists
            # TODO: See if we can log multi-channel images to W&B
            if target.ndim == 3:
                target = target[0]

            target = target.cpu().numpy()
            target -= np.min(target)
            target /= np.max(target)
            visualizations["segmentation"] = wandb_logger.make_image(
                target, data_format="HW", masks=masks
            )

            visualizations = {f"{prefix}/{name}": img for name, img in visualizations.items()}
            wandb_logger.experiment.log({"global_step": self.trainer.global_step, **visualizations})
