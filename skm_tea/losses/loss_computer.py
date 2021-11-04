from meddlr.modeling.loss_computer import LOSS_COMPUTER_REGISTRY, BasicLossComputer, LossComputer

from skm_tea.losses.build import build_loss

EPS = 1e-11
IMAGE_LOSSES = ["l1", "l2", "psnr", "nrmse", "mag_l1", "perp_loss"]
KSPACE_LOSSES = ["k_l1", "k_l1_normalized"]


class SegLossComputer(LossComputer):
    def __init__(self, cfg):
        super().__init__(cfg)
        sigmoid = cfg.MODEL.SEG.ACTIVATION.lower() == "sigmoid"
        include_background = sigmoid or cfg.MODEL.SEG.INCLUDE_BACKGROUND

        self.loss = build_loss(
            cfg.MODEL.SEG.LOSS_NAME,
            include_background=include_background,
            to_onehot_y=False,  # values should be one hot encoded
            sigmoid=sigmoid,
            reduction="mean",
        )

    def __call__(self, input, output):
        target = input["sem_seg"]
        logits = output["sem_seg_logits"]
        loss = self.loss(logits, target)
        return {"loss": loss}


@LOSS_COMPUTER_REGISTRY.register()
class BasicMultiTaskLoss(LossComputer):
    def __init__(self, cfg, tasks):
        super().__init__(cfg)
        recon_loss = cfg.MODEL.RECON_LOSS.NAME
        assert recon_loss in IMAGE_LOSSES

        self.tasks = tasks
        self.recon_computer = BasicLossComputer(cfg)
        self.recon_weight = cfg.MODEL.RECON_LOSS.WEIGHT
        self.seg_computer = SegLossComputer(cfg)
        self.seg_weight = cfg.MODEL.SEG.LOSS_WEIGHT

    def __call__(self, input, output):
        # Recon loss
        if "recon" in self.tasks:
            recon_metrics = {f"recon_{k}": v for k, v in self.recon_computer(input, output).items()}
        else:
            recon_metrics = {"recon_loss": 0.0}

        # Semantic segmentation loss
        if "sem_seg" in self.tasks:
            seg_metrics = {f"sem_seg_{k}": v for k, v in self.seg_computer(input, output).items()}
        else:
            seg_metrics = {"sem_seg_loss": 0.0}

        # TODO: Add bounding box loss.

        metrics = {}
        metrics.update(recon_metrics)
        metrics.update(seg_metrics)

        metrics["loss"] = (
            self.recon_weight * metrics["recon_loss"] + self.seg_weight * metrics["sem_seg_loss"]
        )
        return metrics
