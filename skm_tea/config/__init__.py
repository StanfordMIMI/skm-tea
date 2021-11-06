from meddlr.config.config import set_cfg, set_global_cfg

from skm_tea.config.defaults import _C

__all__ = ["get_cfg"]

set_global_cfg(_C)
set_cfg(_C)


def get_cfg():
    return _C.clone()
