"""Path manager.

DO NOT MOVE THIS FILE.
"""
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from iopath.common.file_io import PathHandler
from meddlr.utils.cluster import Cluster
from meddlr.utils.path import GoogleDriveHandler, URLHandler

from skm_tea.utils import env

# Path to the repository directory.
# TODO: make this cleaner
_REPO_DIR = os.path.join(os.path.dirname(__file__), "../..")


class GeneralPathHandler(PathHandler, ABC):
    PREFIX = ""

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, force: bool = False, **kwargs):
        if force:
            raise ValueError("`force=True` not supported")

        name = path[len(self.PREFIX) :]
        return os.path.join(self._root_dir(), name)

    def _open(self, path: str, mode: str = "r", buffering: int = -1, **kwargs):
        if buffering != -1:
            raise ValueError(f"`buffering={buffering}` not supported")
        return open(self._get_local_path(path), mode, **kwargs)

    def _mkdirs(self, path: str, **kwargs):
        os.makedirs(self._get_local_path(path), exist_ok=True)

    @abstractmethod
    def _root_dir(self) -> str:
        pass


class DataHandler(GeneralPathHandler):
    PREFIX = "data://"

    def _root_dir(self):
        return Cluster.working_cluster().get_path("data_dir")


class ResultsHandler(GeneralPathHandler):
    PREFIX = "results://"

    def _root_dir(self):
        return Cluster.working_cluster().get_path("results_dir")


class CacheHandler(GeneralPathHandler):
    PREFIX = "cache://"

    def _root_dir(self):
        return Cluster.working_cluster().get_path("cache_dir")


class AnnotationsHandler(GeneralPathHandler):
    PREFIX = "ann://"

    def _root_dir(self):
        return os.path.abspath(os.path.join(_REPO_DIR, "annotations"))


class FileSyncHandler(GeneralPathHandler):
    """
    Download data from path and caches them to disk.

    The path will be under ``CACHE_DIR/remote-hostname/path/on/remote/host``
    """

    def __init__(self) -> None:
        super().__init__()
        self.cache_map: Dict[str, str] = {}

    def _root_dir(self):
        return None

    def _cached_path(self, path, cache_dir=None):
        if cache_dir is None:
            cache_dir = Cluster.working_cluster().cache_dir
        remote_path = path[len(self.PREFIX) :]
        remote, path = tuple(remote_path.split(":", maxsplit=1))
        return os.path.join(cache_dir, "rsync", remote, os.path.abspath(path))

    def download(self, remote_path, cache_path):
        return subprocess.run(f"rsync -av {remote_path} {cache_path}", shell=True, check=True)

    def _get_local_path(
        self, path: str, force: bool = False, cache_dir: Optional[str] = None, **kwargs: Any
    ) -> str:
        """
        This implementation downloads the remote resource and caches it locally.
        The resource will only be downloaded if not previously requested.
        """
        remote_path = path[len(self.PREFIX) :]
        cached = self._cached_path(path, cache_dir)
        self._check_kwargs(kwargs)

        if force or not os.path.exists(cached):
            logger = logging.getLogger(__name__)
            logger.info("Downloading {} ...".format(remote_path))
            self.download(remote_path, cached)
            # with file_lock(cached):
            #     if not os.path.exists(cached):
            #         logger.info("Downloading {} ...".format(remote_path))
            #         self.download(remote_path, cached)
            logger.info("Folder {} cached in {}".format(remote_path, cached))
            self.cache_map[path] = cached
        if path not in self.cache_map:
            self.cache_map[path] = cached
        return self.cache_map[path]


class RsyncHandler(FileSyncHandler):
    PREFIX = "rsync://"

    def _cached_path(self, path, cache_dir=None):
        if cache_dir is None:
            cache_dir = Cluster.working_cluster().cache_dir
        remote_path = path[len(self.PREFIX) :]
        remote, path = tuple(remote_path.split(":", maxsplit=1))
        return os.path.join(cache_dir, "rsync", remote, os.path.abspath(path)[1:])

    def download(self, remote_path, cache_path):
        dirpath = (
            os.path.dirname(cache_path)
            if re.match(".*\..*$", os.path.basename(cache_path))
            else cache_path
        )
        os.makedirs(dirpath, exist_ok=True)
        return subprocess.run(f"rsync -av {remote_path} {cache_path}", shell=True, check=True)


class KubernetesHandler(FileSyncHandler):
    PREFIX = "kube://"

    def _cached_path(self, path, cache_dir=None):
        if cache_dir is None:
            cache_dir = Cluster.working_cluster().cache_dir
        remote_path = path[len(self.PREFIX) :]
        remote, path = tuple(remote_path.split(":", maxsplit=1))
        return os.path.join(cache_dir, "kubernetes", remote, os.path.abspath(path)[1:])

    def download(self, remote_path, cache_path):
        cwd = os.getcwd()
        dirpath = (
            os.path.dirname(cache_path)
            if re.match(".*\..*$", os.path.basename(cache_path))
            else cache_path
        )
        os.makedirs(dirpath, exist_ok=True)
        try:
            os.chdir(os.path.expanduser("~"))
            cmd = f"kubectl cp {remote_path} {cache_path}"
            print(cmd)
            out = subprocess.run(cmd, shell=True, check=True)
        finally:
            os.chdir(cwd)
        return out


_path_mgr = env.get_path_manager()
_path_mgr.register_handler(DataHandler())
_path_mgr.register_handler(ResultsHandler())
_path_mgr.register_handler(CacheHandler())
_path_mgr.register_handler(AnnotationsHandler())
_path_mgr.register_handler(KubernetesHandler())
_path_mgr.register_handler(RsyncHandler())
_path_mgr.register_handler(GoogleDriveHandler())
_path_mgr.register_handler(URLHandler(_path_mgr))
