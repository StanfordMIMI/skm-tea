from fvcore.common.registry import Registry
from monai import losses

__all__ = ["LOSS_REGISTRY", "build_loss"]

LOSS_REGISTRY = Registry("LOSS")
LOSS_REGISTRY.__doc__ = """
Registry for losses.

The registered object will be called with `obj(name, **kwargs)`
and expected to return a callable object. In the default config,
the name field is specified by `cfg.MODEL.SEG.LOSS_NAME`.
"""


def build_loss(name, **kwargs):
    if name in losses.__dict__:
        return losses.__dict__[name](**kwargs)
    return LOSS_REGISTRY.get(name)(**kwargs)
