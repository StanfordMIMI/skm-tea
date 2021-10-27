import os

from iopath.common.file_io import PathManager, PathManagerFactory
from ss_recon.utils.cluster import Cluster
from ss_recon.utils.env import is_repro, supports_cupy  # noqa: F401


def get_path_manager(key="skm_tea") -> PathManager:
    return PathManagerFactory.get(key)


def cache_dir() -> str:
    return get_path_manager().get_local_path(
        os.path.join(Cluster.working_cluster().cache_dir, "skm-tea")
    )
