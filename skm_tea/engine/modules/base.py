import logging
import math
import time
from collections import Counter
from typing import Any, Callable, Dict, Sequence, Union

import pytorch_lightning as pl
import torch
from meddlr.data import build_recon_train_loader, build_recon_val_loader
from meddlr.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators
from meddlr.evaluation.testing import flatten_results_dict, print_csv_format
from meddlr.modeling import build_model, initialize_model
from meddlr.modeling.loss_computer import LossComputer
from meddlr.solver import build_lr_scheduler, build_optimizer
from pytorch_lightning.utilities import rank_zero_only
from torch import nn

from skm_tea.engine.trainer import convert_cfg_time_to_iter

__all__ = ["PLModule"]

logger = logging.getLogger(__name__)


class PLModule(pl.LightningModule):
    """Base class for config-driven lightning modules."""

    def __init__(
        self,
        cfg,
        num_parallel: int = 1,
        model: nn.Module = None,
        loss_fn: Callable = None,
        eval_on_cpu: bool = False,
        deployment: bool = False,
    ):
        """
        Args:
            cfg (CfgNode): The config.
            num_parallel (int, optional): The number of GPUs to use.
            model (nn.Module, optional): The model to use.
            loss_fn (Callable, optional): The loss function to use.
            eval_on_cpu (bool, optional): Whether to evaluate on CPU.
            deployment (bool, optional): Whether to use deployment mode.
                This will prevent building data modules used in the PyTorch Lightning
                framework.
        """
        super().__init__()

        # TODO: Configure the right place to do this.
        self.distributed = num_parallel > 1
        self.eval_on_cpu = eval_on_cpu

        if not deployment:
            # Assume these objects must be constructed in this order.
            data_loader = self.train_dataloader(cfg)
            # TODO: Debug this to see if it works for data parallel
            # num_iter_per_epoch = len(data_loader.dataset) / (cfg.SOLVER.TRAIN_BATCH_SIZE * num_parallel)  # noqa: E501
            num_iter_per_epoch = len(data_loader)
            if cfg.DATALOADER.DROP_LAST:
                num_iter_per_epoch = int(num_iter_per_epoch)
            else:
                num_iter_per_epoch = math.ceil(num_iter_per_epoch)
            self.iters_per_epoch = num_iter_per_epoch
            cfg = convert_cfg_time_to_iter(cfg, num_iter_per_epoch)
            del data_loader
        else:
            self.iters_per_epoch = None

        # Build model from config (if required).
        if model is None:
            m_cfg = cfg.clone()
            m_cfg.defrost()
            m_cfg.VIS_PERIOD = 0
            m_cfg.freeze()
            model = self.build_model(m_cfg)
            if m_cfg.MODEL.PARAMETERS.INIT:
                logger.info("Initializing model with parameters in MODEL.INIT")
                initialize_model(model, m_cfg.MODEL.PARAMETERS.INIT)
        if hasattr(model, "vis_period"):
            model.vis_period = -1
        self.vis_period = cfg.VIS_PERIOD
        self.model = model

        self.loss_computer = self.build_loss_computer(cfg) if loss_fn is None else loss_fn
        self.cfg = cfg

        self.std_log = logging.getLogger(__name__)
        self._stage = None

    def train_dataloader(self, cfg=None):
        if cfg is None:
            cfg = self.cfg
        return build_recon_train_loader(cfg, use_ddp=self.use_ddp)

    def val_dataloader(self):
        val_datasets = self.cfg.DATASETS.VAL

        return [
            build_recon_val_loader(self.cfg, dataset, as_test=True, use_ddp=self.use_ddp)
            for dataset in val_datasets
        ]

    def test_dataloader(self):
        test_datasets = self.cfg.DATASETS.TEST

        return [
            build_recon_val_loader(self.cfg, dataset, as_test=True, use_ddp=self.use_ddp)
            for dataset in test_datasets
        ]

    def configure_optimizers(self):
        # Assumes gradient accumulation handled by the trainer.
        # See skm_tea.engine.trainer.PLTrainer
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.SOLVER.GRAD_ACCUM_ITERS = 1
        cfg.freeze()

        opt = build_optimizer(cfg, self.model)
        scheduler = {"scheduler": build_lr_scheduler(self.cfg, opt), "interval": "step"}
        return [opt], [scheduler]

    def training_step(self, inputs: Dict[str, Any], batch_idx):
        assert self.model.training, f"{self._get_name()} model was changed to eval mode!"
        data_time = torch.tensor(self.trainer.profiler.recorded_durations["get_train_batch"][-1])
        profile_times = {
            k: self.trainer.profiler.recorded_durations.get(k, [0.0])
            for k in ["training_step_and_backward", "run_training_batch", "get_train_batch"]
        }
        profile_times = {k: torch.as_tensor(v[-1]) for k, v in profile_times.items() if len(v)}
        data_profiler = inputs.pop("_profiler", {})
        profile_times.update({f"{k}": torch.mean(v) for k, v in data_profiler.items()})

        outputs = self(inputs)

        # Visualize data.
        if (
            self.vis_period is not None
            and self.vis_period > 0
            and self.global_step % self.vis_period == 0
        ):
            self.visualize(inputs, outputs)

        metrics_dict = self._compute_train_metrics(inputs, outputs)
        metrics_dict["data_time"] = data_time
        metrics_dict["train_loss"] = metrics_dict["loss"].clone()
        metrics_dict.update(self.get_learning_rates())
        metrics_dict.update({"profiler": profile_times})

        # result = TrainResult(minimize=metrics_dict.pop("loss"), checkpoint_on=False)
        self.log_dict(
            metrics_dict,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=False,  # don't sync metrics during training
        )
        return metrics_dict["loss"]

    def on_validation_epoch_start(self):
        self._stage = "val"
        self.val_evaluators = [
            self.build_evaluator(self.cfg, dataset_name) for dataset_name in self.cfg.DATASETS.VAL
        ]
        for evaluator in self.val_evaluators:
            evaluator.reset()
        self.val_losses = []

    def validation_step(self, inputs, batch_idx, dataset_idx=0):
        start_time = time.perf_counter()
        outputs = self(inputs)
        inference_time = time.perf_counter() - start_time

        # Assumes all inputs are from the same dataset.
        # dataset_name = inputs["metadata"][0]["dataset_name"]
        evaluator = self.val_evaluators[dataset_idx]

        evaluator.process(inputs, outputs)

        metrics_dict = self._compute_train_metrics(inputs, outputs)
        metrics_dict["batch_eval_time"] = inference_time
        val_loss = metrics_dict["loss"].clone()
        self.val_losses.append(val_loss)

        return {}

    def validation_epoch_end(self, val_logs):
        results = {
            dataset_name: evaluator.evaluate()
            for dataset_name, evaluator in zip(self.cfg.DATASETS.VAL, self.val_evaluators)
        }

        keys = list(results.values())[0].keys()
        flattened_results = flatten_results_dict(results)
        average_results = {
            f"val_{k}": torch.mean(torch.tensor([res[k] for res in results.values()])) for k in keys
        }
        flattened_results.update(average_results)
        flattened_results["val_loss"] = torch.mean(torch.as_tensor(self.val_losses))

        self.std_log.info("Validation Summary")
        for dataset_name, result in results.items():
            self.std_log.info(f"Dataset: {dataset_name}")
            print_csv_format(result)

        self.log_dict(flattened_results)
        del self.val_evaluators
        del self.val_losses
        self._stage = None

    def on_test_epoch_start(self):
        self._stage = "test"
        self.val_evaluators = [
            self.build_evaluator(
                self.cfg, dataset_name, group_by_scan=True, metrics=None, save_data=True
            )
            for dataset_name in self.cfg.DATASETS.TEST
        ]
        for evaluator in self.val_evaluators:
            evaluator.reset()
        self.val_losses = []

    def test_step(self, inputs, batch_idx, dataset_idx=0):
        return self.validation_step(inputs, batch_idx, dataset_idx)

    def test_epoch_end(self, val_logs):
        results = {
            dataset_name: evaluator.evaluate()
            for dataset_name, evaluator in zip(self.cfg.DATASETS.TEST, self.val_evaluators)
        }

        del self.val_evaluators
        del self.val_losses

        self.log_dict(results)
        self._stage = None
        return results

    def get_learning_rates(self) -> Dict[str, float]:
        """Get learning rates for each optimizer."""
        opts = self.optimizers(use_pl_optimizer=False)
        if not isinstance(opts, (list, tuple)):
            opts = [opts]

        lrs = {}
        for idx, optimizer in enumerate(opts):
            # NOTE: some heuristics on what LR to summarize
            # summarize the param group with most parameters
            largest_group = max(len(g["params"]) for g in optimizer.param_groups)

            if largest_group == 1:
                # If all groups have one parameter,
                # then find the most common initial LR, and use it for summary
                lr_count = Counter([g["lr"] for g in optimizer.param_groups])
                lr = lr_count.most_common()[0][0]
                for i, g in enumerate(optimizer.param_groups):
                    if g["lr"] == lr:
                        best_param_group_id = i
                        break
            else:
                for i, g in enumerate(optimizer.param_groups):
                    if len(g["params"]) == largest_group:
                        best_param_group_id = i
                        break
            lrs[f"lr/opt_{idx}"] = optimizer.param_groups[best_param_group_id]["lr"]

        if len(lrs) == 1:
            lrs = {"lr": list(lrs.values())[0]}
        return lrs

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @rank_zero_only
    def visualize(self, inputs, outputs, prefix="train", channels=None):
        """Handle visualization."""
        pass

    def build_evaluator(
        self, cfg, dataset_name, **kwargs
    ) -> Union[DatasetEvaluator, Sequence[DatasetEvaluator], DatasetEvaluators]:
        """
        Returns:
            DatasetEvaluator

        It is not implemented by default.
        """
        pass

    def build_loss_computer(self, cfg) -> LossComputer:
        pass

    def _compute_train_metrics(self, inputs, outputs):
        """Wrapper for computing train metrics per batch.

        Returns:
            dict[str, Tensor]: A dictionary with keywords (shown below).
                All other metrics are scalar tensors that should be logged.

                Keywords:
                    * "loss": The tensor to call `.backwards()` on

        """
        loss_dict = {k: v for k, v in outputs.items() if "loss" in k}
        running_loss = sum(loss_dict.values())

        if hasattr(self, "loss_computer") and self.loss_computer:
            loss_comp_dict = self.loss_computer(inputs, outputs)
            if "loss" in loss_comp_dict:
                running_loss += loss_comp_dict["loss"]
            else:
                running_loss += sum(v for k, v in loss_comp_dict.items() if "loss" in k)
            loss_dict.update(loss_comp_dict)

        metrics_dict = loss_dict
        metrics_dict.update(
            self.metrics_computer(outputs)
            if hasattr(self, "metrics_computer") and self.metrics_computer
            else {}
        )
        metrics_dict["loss"] = running_loss

        # Update any non-tensor values.
        tensor_scalars = {
            k: torch.tensor(v) for k, v in metrics_dict.items() if not isinstance(v, torch.Tensor)
        }
        metrics_dict.update(tensor_scalars)

        return metrics_dict
