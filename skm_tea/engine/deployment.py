import os
import warnings
from typing import Union

import meddlr.config.util as config_util
import torch
from meddlr.config import CfgNode

from skm_tea.config import get_cfg
from skm_tea.engine.modules import SkmTeaModule, SkmTeaSemSegModule
from skm_tea.utils import env

__all__ = ["build_deployment_model"]


def build_deployment_model(
    cfg_or_file: Union[str, os.PathLike, CfgNode],
    weights_file: Union[str, os.PathLike] = None,
    unpack_model: bool = False,
    strict: bool = True,
    force_download: bool = False,
):
    """Build model and optionally load in weights.

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

    if weights_file is None:
        weights_file = cfg.MODEL.WEIGHTS
    if not weights_file:
        return model.model if unpack_model else model

    weights_file = path_manager.get_local_path(weights_file, force=force_download)
    weights = torch.load(weights_file)
    if "state_dict" in weights:
        weights = weights["state_dict"]

    # TODO (arjundd): Add method to check what device the weights are on
    # and to move model accordingly.

    model.load_state_dict(weights, strict=strict)
    if unpack_model:
        model = model.model
    return model
