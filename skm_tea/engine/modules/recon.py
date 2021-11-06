import torch
from meddlr.evaluation import ReconEvaluator
from meddlr.ops import complex as cplx
from pytorch_lightning.utilities import rank_zero_only

from skm_tea.engine.modules.base import PLModule
from skm_tea.utils import visualizer as vis

__all__ = ["ReconModule"]


class ReconModule(PLModule):
    """"""

    @torch.no_grad()
    @rank_zero_only
    def visualize(self, inputs, outputs, prefix="train", channels=None):
        """Handle visualization."""
        if not hasattr(self.logger, "log_images"):
            raise ValueError("self.logger does not support logging images")

        zf_image = outputs["zf_image"][0].squeeze()
        pred = outputs["pred"][0].squeeze()
        target = outputs["target"][0].squeeze()
        if cplx.is_complex(inputs["kspace"]):
            kspace = inputs["kspace"][0, ..., 0]  # mask for first image and first coil
        else:
            kspace = inputs["kspace"][0, ..., 0, :]  # mask for first image and first coil

        visualizations = vis.draw_reconstructions(
            [zf_image, pred], target, kspace, channels=channels
        )
        visualizations = {f"{prefix}/{name}": img for name, img in visualizations.items()}
        self.logger.log_images(visualizations, step=self.trainer.global_step, data_format="CHW")

    def build_evaluator(self, cfg, dataset_name, **kwargs) -> ReconEvaluator:
        """
        Returns:
            DatasetEvaluator

        It is not implemented by default.
        """
        # Aggregating scans with small batch sizes results in volume dimensions
        # smaller than window size of kernel used to calculate SSIM
        # Turn off aggregation when running sanity check or fast_dev_run
        aggregate_scans = not self.trainer.sanity_checking and not self.trainer.fast_dev_run
        metrics = kwargs.pop("metrics", cfg.TEST.VAL_METRICS.RECON)
        flush_period = kwargs.pop("flush_period", cfg.TEST.FLUSH_PERIOD)
        to_cpu = kwargs.pop("to_cpu", self.eval_on_cpu)

        return ReconEvaluator(
            dataset_name,
            cfg,
            distributed=self.use_ddp,
            sync_outputs=False,
            aggregate_scans=aggregate_scans,
            metrics=metrics,
            flush_period=flush_period,
            to_cpu=to_cpu,
            eval_in_process=True,
            **kwargs,
        )
