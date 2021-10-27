import os
import re
from typing import List

from skm_tea.utils import env

_PATH_MANAGER = env.get_path_manager()


def format_exp_version(dir_path, new_version=True, mkdirs=False, force=False):
    """Adds experiment version to the directory path. Returns local path.

    If `os.path.basename(dir_path)` starts with 'version', assume the version
    has already been formatted.

    Args:
        dir_path (str): The directory path corresponding to the version.
        force (bool, optional): If `True` force adds version even if 'version'
            is part of basename.

    Returns:
        str: The formatted dirpath
    """
    dir_path = _PATH_MANAGER.get_local_path(dir_path)
    if not os.path.isdir(dir_path):
        return os.path.join(dir_path, "version_001")
    if not force and re.match("^version_[0-9]*", os.path.basename(dir_path)):
        return dir_path
    version_dir, version_num = _find_latest_version_dir(dir_path)
    if new_version:
        version_num += 1
        version_dir = f"version_{version_num:03d}"
    version_dirpath = os.path.join(dir_path, version_dir)
    if mkdirs:
        _PATH_MANAGER.mkdirs(version_dirpath)
    return version_dirpath


def find_experiment_dirs(dirpath, completed=True) -> List[str]:
    """Find all experiment directories under the `dirpath`.

    Args:
        dirpath (str): The directory under which to search.
        completed (bool, optional): If `True`, filter directories where runs
            are completed.

    Returns:
        exp_dirs (List[str]): A list of experiment directories.
    """

    def _find_exp_dirs(_dirpath):
        # Directories with "config.yaml" are considered experiment directories.
        if os.path.isfile(os.path.join(_dirpath, "config.yaml")):
            return [_dirpath]
        # Directories with no more subdirectories do not have a path.
        subfiles = [os.path.join(_dirpath, x) for x in os.listdir(_dirpath)]
        subdirs = [x for x in subfiles if os.path.isdir(x)]
        if len(subdirs) == 0:
            return []
        exp_dirs = []
        for dp in subdirs:
            exp_dirs.extend(_find_exp_dirs(dp))
        return exp_dirs

    dirpath = _PATH_MANAGER.get_local_path(dirpath)
    exp_dirs = _find_exp_dirs(dirpath)
    if completed:
        exp_dirs = [x for x in exp_dirs if os.path.isfile(os.path.join(x, "model_final.pth"))]
    return exp_dirs


def _find_latest_version_dir(dir_path):
    version_dirs = [
        (x, int(x.split("_")[1])) for x in os.listdir(dir_path) if re.match("^version_[0-9]*", x)
    ]
    if len(version_dirs) == 0:
        version_dir, version_num = None, 0
    else:
        version_dirs = sorted(version_dirs, key=lambda x: x[1])
        version_dir, version_num = version_dirs[-1]
    return version_dir, version_num
