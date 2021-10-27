import torch
from monai.losses import DiceLoss

from skm_tea.losses.build import LOSS_REGISTRY

__all__ = ["FlattenedDiceLoss"]


@LOSS_REGISTRY.register()
class FlattenedDiceLoss(DiceLoss):
    """Compute Dice loss after flattening all classes.

    Instead of computing average Dice across different classes, this module
    flattens all classes into a single class. This can cause optimization
    issues when classes are imbalanced.
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input.reshape(input.shape[0], 1, -1)
        target = target.reshape(target.shape[0], 1, -1)
        return super().forward(input, target)
