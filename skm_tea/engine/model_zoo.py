import os
import warnings
from typing import Union

import meddlr.config.util as config_util
import torch
from meddlr.config import CfgNode

from skm_tea.config import get_cfg
from skm_tea.engine.modules import SkmTeaModule, SkmTeaSemSegModule
from skm_tea.utils import env

__all__ = ["get_model_from_zoo"]


def get_model_from_zoo(
    cfg_or_file: Union[str, os.PathLike, CfgNode],
    weights_path: Union[str, os.PathLike] = None,
    unpack_model: bool = False,
    strict: bool = True,
    ignore_shape_mismatch: bool = False,
    force_download: bool = False,
) -> torch.nn.Module:
    """Get model from zoo and optionally load in weights.

    This function is designed for distributing models for use.
    It builds the model from a configuration and optionally loads in pre-trained weights.

    Pre-trained weights can either be specified by the ``weights_file`` argument
    or by setting the config option ``cfg.MODEL.WEIGHTS``. If neither is specified,
    the model is initialized randomly. If ``weights_file`` is an empty string,
    ``cfg.MODEL.WEIGHTS`` will also be ignored, and the model will be initialized randomly.

    Args:
        cfg_or_file (str | path-like | CfgNode): The config (or file).
            The file is recommended to check for dependency conflicts.
        weights_file (str | path-like, optional): The weights file to load.
            This can also be specified in the ``cfg.MODEL.WEIGHTS`` config
            field. If neither are provided, the uninitialized model will be returned.
        unpack_model (bool, optional): If ``True``, this model will be unpacked
            from the wrapper PyTorch Lightnining modules.
        strict (bool, optional): Strict option to pass to ``load_state_dict``.
        force_download (bool, optional): Force download of model config/weights
            if downloading the model.

    Returns:
        torch.nn.Module: The model loaded with pre-trained weights.
            Note this model will be unpacked from the PyTorch Lightnining modules.
    """
    path_manager = env.get_path_manager()

    if not isinstance(cfg_or_file, CfgNode):
        cfg_file = path_manager.get_local_path(cfg_or_file, force=force_download)
        failed_deps = config_util.check_dependencies(cfg_file, return_failed_deps=True)
        if len(failed_deps) > 0:
            warning_msg = (
                f"Some dependenices are not met. "
                f"This may result in some issues with model construction or weight loading. "
                f"Unmet dependencies: {failed_deps}"
            )
            warnings.warn(warning_msg)
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)

    pl_module = SkmTeaModule if "recon" in cfg.MODEL.TASKS else SkmTeaSemSegModule
    model = pl_module(cfg, deployment=True)

    if weights_path is None:
        weights_path = cfg.MODEL.WEIGHTS
    if not weights_path:
        return model.model if unpack_model else model

    weights_path = path_manager.get_local_path(weights_path, force=force_download)
    model = load_weights(
        model, weights_path, strict=strict, ignore_shape_mismatch=ignore_shape_mismatch
    )
    if unpack_model:
        model = model.model
    return model


def load_weights(
    model: torch.nn.Module,
    weights_path: Union[str, os.PathLike],
    strict: bool = True,
    ignore_shape_mismatch: bool = False,
    force_download: bool = False,
) -> torch.nn.Module:
    """Load model from checkpoint.

    This function is designed for distributing models for use.
    It loads in pre-trained weights.

    Pre-trained weights can either be specified by the ``weights_file`` argument
    or by setting the config option ``cfg.MODEL.WEIGHTS``. If neither is specified,
    the model is initialized randomly. If ``weights_file`` is an empty string,
    ``cfg.MODEL.WEIGHTS`` will also be ignored, and the model will be initialized randomly.

    Args:
        model (torch.nn.Module): The model to load weights into.
        checkpoint_path (str | path-like): The checkpoint file to load.
        strict (bool, optional): Strict option to pass to ``load_state_dict``.
        unpack_model (bool, optional): If ``True``, this model will be unpacked
            from the wrapper PyTorch Lightnining modules.

    Returns:
        torch.nn.Module: The model loaded with pre-trained weights.
            Note this model will be unpacked from the PyTorch Lightnining modules.
    """
    path_manager = env.get_path_manager()

    weights_path = path_manager.get_local_path(weights_path, force=force_download)
    weights = torch.load(weights_path)
    if "state_dict" in weights:
        weights = weights["state_dict"]

    # TODO (arjundd): Add method to check what device the weights are on
    # and to move model accordingly.

    if ignore_shape_mismatch:
        params_shape_mismatch = _find_mismatch_sizes(model, weights)
        if len(params_shape_mismatch) > 0:
            mismatched_params_str = "".join("\t- {}\n".format(x) for x in params_shape_mismatch)
            warnings.warn(
                "Shape mismatch found for some parameters. Ignoring these weights:\n{}".format(
                    mismatched_params_str
                )
            )
            for p in params_shape_mismatch:
                weights.pop(p)
            strict = False

    model.load_state_dict(weights, strict=strict)
    return model


def _find_mismatch_sizes(model: torch.nn.Module, state_dict):
    """Finds the keys in the state_dict that are different from the model.

    Args:
        model (torch.nn.Module): The model to load weights into.
        state_dict (dict): The state_dict to load.

    Returns:
        list[str]: The list of keys that are different from the model.
    """
    params_shape_mismatch = set()
    for source in [model.state_dict().items(), model.named_parameters()]:
        for name, param in source:
            if name not in state_dict:
                continue
            if param.shape != state_dict[name].shape:
                params_shape_mismatch |= {name}

    return list(params_shape_mismatch)
