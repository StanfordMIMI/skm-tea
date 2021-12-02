"""Wrapper for MONAI networks."""

import inspect
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union

import torch.nn as nn
from meddlr.modeling.meta_arch.build import META_ARCH_REGISTRY
from monai.networks.nets import UNet, VNet
from monai.networks.nets.dynunet import DynUNet

__all__ = ["VNetMONAI", "UNetMONAI", "DynUNetMONAI"]


def cfg_to_kwargs(cfg, klass, arg_cfg_map=None, skip_args=None):
    """Convert config into keyword arguments MONAI network.

    Args:
        cfg (CfgNode): The config.
        klass: The MONAI model class to instantiate.
        arg_cfg_map (Dict[str, str], optional): Map from init param name to config param name.
        skip_args (bool, optional): Arguments to skip.
    """
    argspec = inspect.getfullargspec(klass)
    args = argspec.args
    defaults = argspec.defaults

    # Map arguments to defaults
    defaults = {arg: default for arg, default in zip(args[-len(defaults) :], defaults)}

    if "self" in args:
        args.remove("self")

    kwargs = {}
    for arg in args:
        if skip_args and arg in skip_args:
            continue
        cfg_key = arg_cfg_map[arg] if arg_cfg_map and arg in arg_cfg_map else arg.upper()
        value = cfg.get(cfg_key, defaults.get(arg, None))

        # If value is still None, try taking the default
        if not value:
            if arg in defaults:
                value = defaults[arg]
            else:
                raise ValueError(f"{cfg_key} must be specified")

        kwargs[arg] = value

    return kwargs


def _parse_activation(activation: Union[str, Tuple[str, Dict]]):
    if isinstance(activation, str):
        return activation

    if len(activation) == 1:
        return activation[0]

    return activation


class _MONAINetwork(nn.Module, ABC):
    """Modules that are wrappers around MONAI networks should inherit from this class."""

    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()
        self.cfg = cfg
        self.spatial_dims = 2

        args = {
            "spatial_dims": self.spatial_dims,
            "in_channels": in_channels,
            "out_channels": out_channels,
        }
        self.net = self.build_model(**args)

    @abstractmethod
    def build_model(self, **kwargs) -> nn.Module:
        pass

    def forward(self, *args, **kwargs):
        # TODO: Do some parsing here.
        # For now assumes that the input is the image to toss into the network
        return self.net(*args, **kwargs)


@META_ARCH_REGISTRY.register()
class VNetMONAI(_MONAINetwork):
    """V-Net"""

    ALIASES = ["VNET_MONAI"]
    CONFIG_KEY = "VNET_MONAI"

    def build_model(self, **kwargs):
        additional_kwargs = cfg_to_kwargs(
            self.cfg, VNet, arg_cfg_map={"act": "ACTIVATION"}, skip_args=kwargs.keys()
        )
        additional_kwargs["act"] = _parse_activation(additional_kwargs["act"])
        kwargs.update(additional_kwargs)
        return VNet(**kwargs)


@META_ARCH_REGISTRY.register()
class UNetMONAI(_MONAINetwork):
    ALIASES = ["UNET_MONAI"]
    CONFIG_KEY = "UNET_MONAI"

    def build_model(self, **kwargs):
        if "dimensions" not in kwargs:
            kwargs["dimensions"] = kwargs.pop("spatial_dims", None)
        additional_kwargs = cfg_to_kwargs(
            self.cfg, UNet, arg_cfg_map={"act": "ACTIVATION"}, skip_args=kwargs.keys()
        )
        additional_kwargs["act"] = _parse_activation(additional_kwargs["act"])
        kwargs.update(additional_kwargs)
        return UNet(**kwargs)


@META_ARCH_REGISTRY.register()
class DynUNetMONAI(_MONAINetwork):
    ALIASES = ["DYNUNET_MONAI"]
    CONFIG_KEY = "DYNUNET_MONAI"

    def build_model(self, **kwargs):
        upsample_kernel_size = kwargs.get("upsample_kernel_size", self.cfg.UPSAMPLE_KERNEL_SIZE)
        if isinstance(upsample_kernel_size, int):
            upsample_kernel_size = (upsample_kernel_size,)
        if len(upsample_kernel_size) == 1:
            upsample_kernel_size = (upsample_kernel_size[0],) * (len(self.cfg.KERNEL_SIZE))
        kwargs["upsample_kernel_size"] = upsample_kernel_size

        additional_kwargs = cfg_to_kwargs(self.cfg, DynUNet, skip_args=kwargs.keys())
        kwargs.update(additional_kwargs)
        return DynUNet(**kwargs)
