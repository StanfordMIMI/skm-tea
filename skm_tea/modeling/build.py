import inspect
import logging

from meddlr.modeling.meta_arch import META_ARCH_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["build_model"]


def build_model(cfg, in_channels=None, out_channels=None, **kwargs):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if "/" in meta_arch:
        meta_arch, build_arch = tuple(meta_arch.split("/", 1))
    else:
        build_arch = None
    klass = META_ARCH_REGISTRY.get(meta_arch)

    channel_args = {"in_channels": in_channels, "out_channels": out_channels}
    sig = inspect.signature(klass)
    for k, v in channel_args.items():
        if k in sig.parameters:
            kwargs[k] = v

    build_cfg = cfg
    if build_arch:
        for k in build_arch.split("."):
            build_cfg = build_cfg[k]

    # TODO: Replace with a Mixin
    if hasattr(klass, "from_config"):
        return klass(build_cfg, **kwargs)

    return klass(build_cfg, **kwargs)


def get_model_cfg(cfg):
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    klass = META_ARCH_REGISTRY.get(meta_arch)
    if hasattr(klass, "CONFIG_KEY"):
        return cfg.MODEL[klass.CONFIG_KEY].clone()
    raise ValueError(f"Cannot get model config from class {klass}")
