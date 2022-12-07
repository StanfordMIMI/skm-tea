import logging
import os
from copy import deepcopy
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Union

import dosma as dm
import h5py
import meddlr.ops as oF
import nibabel as nib
import numpy as np
import torch
from meddlr.data.data_utils import HDF5Manager, collect_mask
from meddlr.data.slice_dataset import SliceData
from meddlr.utils import profiler
from tqdm.auto import tqdm

from skm_tea.data.register import seg_categories_to_idxs
from skm_tea.data.transform import qDESSDataTransform
from skm_tea.utils import env

__all__ = ["SkmTeaRawDataset", "SkmTeaDicomDataset"]
logger = logging.getLogger(__name__)


class SkmTeaRawDataset(SliceData):
    """The dataset to use for the SKM-TEA Raw Data Track.

    This dataset is used to load inputs and labels for the SKM-TEA Raw Data Track.
    Currently, reconstruction and semantic segmentation tasks are supported.

    Examples::

        from meddlr.data import DatasetRegistry
        from skm_tea.data import SkmTeaRawDataset

        # Load the SKM-TEA v1 train split.
        dataset_dicts = DatasetRegistry.get("skmtea_v1_train")

        # Build dataset for Raw Data Track reconstruction for echo 1.
        transform = qDESSDataTransform(...)
        dataset = SkmTeaRawDataset(
            dataset_dicts, transform=transform, split="train", tasks=["recon"], echo_type="echo1"
        )
        example0 = dataset[0]

        # Build dataset Raw Data Track  semantic segmentation on echo 1.
        transform = qDESSDataTransform(...)
        dataset = SkmTeaRawDataset(
            dataset_dicts, transform=transform, split="train", tasks=["sem_seg"], echo_type="echo1"
        )

    Attributes:
        split (str): The dataset split. One of ``['train', 'val', 'test']``.
        echo_type (str): The type of echo to load. One of
            ``['echo1', 'echo2', 'echo1+echo2', 'echo1-echo2-mc']``.
        tasks (Sequence[str]): The tasks to generate data for.
        use_segmentation (bool, optional): If ``True``, segmentation data will be loaded.
        use_detection (bool, optional): If ``True``, detection data will be loaded.
        use_recon (bool, optional): If ``True``, reconstruction data will be loaded.
        seg_classes (Sequence[Union[str, Sequence[str]]], optional): Classes to use
            for segmentation.
        seg_idxs (Sequence[Union[str, Sequence[str]]], optional): Categories (as indices)
            to use for segmentation.
        keys_to_load (Collection[str], optional): Extra keys to load from HDF5 file.
        stats (dict of dicts): Statistics in form ``key->scan_id->echo->metric``.
            For example, for the mean of echo1 volume for scan MTR_001's ``'target'`` key
            can be written as ``self.stats['target']['MTR_001']['echo1']['mean']``.
            For Raw Data Track, statistics are taken over the magnitude target recon image.
        normalization (Dict[str, Union[float, str]]): The numbers (float) or stats values (str)
            to normalize data volume by.
        cache_files (bool): If ``True``, keeps HDF5 files open to eliminate file opening time.
    """

    _ALIASES = ("RawDataTrack", "SkmTeaRawDataTrack", "SkmTeaRawDataTrackDataset", "qDESSDataset")
    _REQUIRED_METADATA = ()

    # TODO: Clean up - get rid of set of keys. All dataset dict properties
    # should be loaded into each example.
    _DD_KEYS = [
        "recon_file",
        "image_file",
        "matrix_shape",
        "scan_id",
        "subject_id",
        "voxel_spacing",
        "num_coils",
        "orientation",
        "gw_corr_mask_file",
    ]
    _METADATA_KEYS = ["matrix_shape", "scan_id", "subject_id", "voxel_spacing", "acq_shape"]
    _stats_file = os.path.join(env.cache_dir(), "stats", "raw-data-stats.pt")

    def __init__(
        self,
        dataset_dicts: List[Dict],
        transform: Callable,
        split: str,
        keys: Optional[Dict[str, str]] = None,
        include_metadata: bool = False,
        tasks: Sequence[str] = ("recon",),
        seg_classes: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        keys_to_load: Collection[str] = None,
        echo_type: str = "echo1",
        cache_files: bool = False,
        normalization: Optional[str] = "_default",
    ):
        """
        Args:
            dataset_dicts: A list of dictionaries representing the dataset.
                Each dictionary corresponds to a single scan.
            transform: The transform to apply to the data.
                Typically, this is of type :class:`qDESSDataTransform`.
            split: The dataset split. One of ``['train', 'val', 'test']``.
            keys: A dictionary mapping dataset keys to HDF5 file keys.
                See :class:`meddlr.data.SliceDataset` for more information.
            include_metadata: Whether to return metadata when returning the example
                in ``__getitem__``.
            tasks: The task(s) to return data for. This dataset currently supports tasks:
                * ``'recon'``: MRI reconstruction
                * ``'sem_seg'``: Semantic segmentation
            seg_classes: The integer id(s) for segmentation classes to load.
                This must be specified if ``'sem_seg' in tasks``.
                See :obj:`skm_tea.data.register.SKMTEA_SEGMENTATION_CATEGORIES` for
                the mapping from class -> id.
            keys_to_load: A list of keys to load from the HDF5 file.
                By default, these keys are determined by the task.
            echo_type: The type of echo to load. One of:
                * ``'echo1'``: Echo 1
                * ``'echo2'``: Echo 2
                * ``'echo1+echo2'``: Echo 1 and echo 2, where echos are different examples
                * ``'echo1-echo2-mc'``: Echo 1 and echo 2, where echos are multiple channels
                * ``'rss'``: The root sum-of-squares of the echoes.
                  This type is only valid for downstream tasks (e.g. segmentation, detection).
            cache_files: Whether to keep the HDF5 files open after opening once.
                This may be useful for faster loading times, but can cause issues on external
                mounted drives (e.g. network attached storage).
            normalization: The normalization method to use.
                Defaults to scaling by the standard devation volume.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("`split` must be one of '['train', 'val', 'test']")
        if not isinstance(transform, qDESSDataTransform):
            raise TypeError("`transform` must be a qDESSDataTransform")

        self.split = split

        if echo_type not in ["echo1", "echo2", "echo1+echo2", "echo1-echo2-mc"]:
            raise ValueError(f"Unknown echo_type {echo_type}")
        self.echo_type = echo_type

        self.use_segmentation = "sem_seg" in tasks
        if self.use_segmentation:
            assert seg_classes is not None, "Seg classes must be specified"
            self.seg_idxs = seg_categories_to_idxs(seg_classes)

        self.use_detection = "detection" in tasks
        self.use_recon = "recon" in tasks

        if self.use_detection:
            raise ValueError("Detection is not yet supported.")

        self.tasks = tasks
        self.keys_to_load = keys_to_load

        self.stats = self._build_stats(dataset_dicts)
        self.cache_files = cache_files

        if normalization == "_default":
            normalization = {"bias": 0.0, "scale": "target/{scan_id}/total/std"}
        elif normalization is None:
            normalization = {"bias": 0.0, "scale": 1.0}
        self.normalization = normalization

        super().__init__(dataset_dicts, transform, keys, include_metadata)

    def _build_stats(
        self, dataset_dicts: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """Build statistics dictionary.

        The ``min``, ``max``, ``mean``, ``std``, and ``median``
        are computed for the magnitude of the target (ground-truth) reconstruction
        volume for each scan.
        These values are computed for echo1, echo2, echo1-echo2 combined (i.e. ``total``),
        and root-sum-of-squares (rss).

        This function also caches these values to avoid recomputing them on each run.

        Args:
            dataset_dicts: A list of dictionaries representing the dataset.
                Each dictionary corresponds to a single scan.

        Returns:
            dict: Statistics in form ``key->scan_id->echo->stat``.
                For example ``stats['target']['MTR_001']['echo1']['mean']``
                would give the mean of the absolute value of the target reconstruction
                of echo1 for scan ``MTR_001``.
        """
        funcs = {"min": np.min, "max": np.max, "mean": np.mean, "std": np.std, "median": np.median}
        stats_file = self._stats_file

        stats = {}
        if os.path.isfile(stats_file):
            logger.info(f"Loading stats from {stats_file}")
            stats = torch.load(stats_file)

        logger.info("Building statistics...")
        target_stats = stats.get("target", {})
        modified = False
        for dd in tqdm(dataset_dicts, desc="Building dataset stats"):
            scan_id = dd["scan_id"]
            if scan_id in target_stats:
                continue
            with h5py.File(dd["recon_file"], "r") as f:
                vol = np.abs(f["target"][()])
            input_types = {
                "total": vol,
                "echo1": vol[..., 0, :],
                "echo2": vol[..., 1, :],
                "rss": np.sqrt(np.sum(vol**2, axis=-2)),
            }
            scan_stats = {
                k: {fname: func(v) for fname, func in funcs.items()} for k, v in input_types.items()
            }
            target_stats[dd["scan_id"]] = scan_stats
            modified = True

        stats = {"target": target_stats}
        if modified:
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            logger.info(f"Saving statistics to {stats_file}...")
            torch.save(stats, stats_file)

        # This is modifying the stats dictionaries for each mtr_id in-place.
        stats_dict: Dict[str, Dict[str, Dict[str, float]]]
        for stats_dict in target_stats.values():
            stats_dict.update(
                {
                    "echo1-echo2-mc": {
                        k: np.stack([stats_dict["echo1"][k], stats_dict["echo2"][k]], axis=-1)
                        for k in funcs
                    }
                }
            )
        stats = {"target": target_stats}
        return stats

    def _init_examples(self, dataset_dicts: List[Dict[str, Any]], slice_dim: int = 0):
        """Convert dataset dicts into examples.

        Dataset dictionaries have a 1:1 correspondence with each scan. This function
        creates multiple examples from a single scan dictionary, where each example
        corresponds to a slice of the scan.

        Args:
            dataset_dicts: A list of dictionaries representing the dataset.
                Each dictionary corresponds to a single scan.
            slice_dim: The dimension to slice the scan along.
                0 - axial, 1 - coronal, 2 - sagittal.

        Returns:
            List[Dict[str, Any]]: The list of examples. Each dictionary contains:
                * ``'scan_id'`` (str): The scan ID.
                * ``'slice_id'`` (int): The zero-indexed slice ID.
                * ``'acq_shape'`` (Tuple[int, int]): The 2D shape of the acquisition matrix.
                  This is the shape before the k-space has been zero-padding.
                  The acq_shape is needed to generate undersampling masks that correspond to
                  the true acquisition shape.
                * ``'inplane_shape'`` (Tuple[int, int]): The 2D shape of the reconstructed image.
                  This is the shape after the k-space has been zero-padded.
        """
        examples = []
        echo_vals = {
            "echo1": (0,),
            "echo2": (1,),
            "echo1+echo2": (0, 1),
            "echo1-echo2-mc": (slice(None),),
        }[self.echo_type]

        for dd in dataset_dicts:
            num_slices = dd["matrix_shape"][slice_dim]
            inplane_shape = tuple(s for d, s in enumerate(dd["matrix_shape"][:3]) if d != slice_dim)
            # TODO: Update this to get acq_shape from the metadata directly.
            # This works for now because all data have padding of 40/40 in the kz direction.
            acq_shape = (512, 416, dd["matrix_shape"][2] - 80)
            acq_shape = tuple(dim for idx, dim in enumerate(acq_shape) if idx != slice_dim)

            dd_examples = []
            for slice_id in range(num_slices):
                # TODO: Change when multiple echos supported
                base_ex = {"slice_id": slice_id, "acq_shape": acq_shape}
                for echo in echo_vals:
                    ex = deepcopy(base_ex)
                    ex["echo"] = echo
                    if self.use_detection:
                        raise ValueError("Detection is not supported")
                        # Format boxes into 2D boxes in XYWH_ABS format
                        # ex["annotations"] = slice_bboxes(
                        #     dd["annotations"], slice_id + 1, dim, transpose=True, copy=True
                        # )
                    ex.update({k: dd[k] for k in self._DD_KEYS})
                    ex.update({k: tuple(v) for k, v in ex.items() if isinstance(v, list)})
                    ex["inplane_shape"] = inplane_shape
                    dd_examples.append(ex)

            examples.extend(dd_examples)

        return examples

    def _load_files(self, file_keys: Optional[Sequence[str]] = None):
        """Initialize file manager.

        Args:
            file_keys: A list of keys that map to HDF5 file paths.
                If ``None``, files are loaded based on the specified tasks
                (i.e. ``self.tasks``).
        """
        if file_keys:
            files = {dd[k] for k in file_keys for dd in self.examples}
        else:
            files = {dd["recon_file"] for dd in self.examples}
            if self.use_segmentation:
                files = files | {dd["image_file"] for dd in self.examples}
        self.file_manager = HDF5Manager(files, cache=self.cache_files)

    def _get_stats(self, fmt_str: str = None, **kwargs):
        def _parse_stats(_fmt_str):
            stats = self.stats
            for skey in _fmt_str.format(**kwargs).split("/"):
                stats = stats[skey]
                val = stats
            return val

        if fmt_str is None and len(kwargs) == 0:
            return self.stats

        if fmt_str is not None:
            return _parse_stats(fmt_str)

        affine_params = {}
        for k, v_default in (("bias", 0.0), ("scale", 1.0)):
            val = self.normalization.get(k, v_default)
            if isinstance(val, str):
                val = _parse_stats(val)
            affine_params[k] = val
        return affine_params

    @profiler.time_profile()
    def _load_data(self, example: Dict[str, Any], idx: int):
        """Loads matrix data into the example."""
        _READ_DATA = "read_data"
        timer = profiler.get_timer()

        slice_id = example["slice_id"]
        scan_id = example["scan_id"]
        output = {}

        # Load recon info.
        fp = example.pop("recon_file")
        if not hasattr(self, "file_manager"):
            self._load_files()

        # Determine orientation
        orientation = example["orientation"]
        voxel_spacing = example["voxel_spacing"]
        affine = dm.to_affine(orientation, spacing=voxel_spacing)

        timer.start(_READ_DATA)
        with self.file_manager.yield_file(fp) as data:
            output["target"] = data[self.mapping["target"]][slice_id, ..., example["echo"], :]
            if self.use_recon:
                output["kspace"] = data[self.mapping["kspace"]][slice_id, ..., example["echo"], :]
                output["maps"] = data[self.mapping["maps"]][slice_id]
                if self.split == "test":
                    acc = self.transform._subsampler.mask_func.accelerations
                    assert len(acc) == 1
                    acc = acc[0]
                    output["mask"] = torch.from_numpy(data[f"masks/poisson_{float(acc)}x"][()])

        if self.echo_type == "echo1-echo2-mc":
            output["target"] = output["target"][..., 0]
            if "kspace" in output:
                ndim = output["kspace"].ndim
                output["kspace"] = output["kspace"].transpose(
                    tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
                )

        if self.use_segmentation:
            mask_file = example.pop("gw_corr_mask_file")
            # TODO: Loading from nifti (even with memmap) is still pretty slow.
            sem_seg = nib.load(mask_file).dataobj[slice_id]
            # with self.file_manager.yield_file(example.pop("image_file")) as data:
            #     sem_seg = data["seg"][slice_id]
            sem_seg = oF.categorical_to_one_hot(
                sem_seg, channel_dim=-1, background=0, num_categories=6
            )
            sem_seg = collect_mask(sem_seg, self.seg_idxs, out_channel_first=True)
            output["sem_seg"] = sem_seg.astype(np.float32)
        timer.stop(_READ_DATA)

        # Do transformation before annotations are converted to 2D instances.
        echo_kind = self.echo_type
        if echo_kind == "echo1+echo2":
            echo_kind = f"echo{example['echo']+1}"
        affine_params = self._get_stats(scan_id=scan_id, echo_kind=echo_kind)
        output = self.transform(
            output,
            scan_id=example["scan_id"],
            slice_id=slice_id,
            acq_shape=example.get("acq_shape", None),
            **affine_params,
        )

        output["stats"] = {
            "target": {
                "vol_mean": self._get_stats(f"target/{scan_id}/{echo_kind}/mean"),
                "vol_std": self._get_stats(f"target/{scan_id}/{echo_kind}/std"),
            }
        }

        if self.use_detection and "annotations" in example:
            img_size = example["matrix_shape"][1:]  # noqa: F841
            # output["instances"] = annotations_to_instances2d(example.pop("annotations"), img_size)

        if self._include_metadata:
            output["metadata"] = {
                "orientation": orientation,
                "voxel_spacing": voxel_spacing,
                "affine": affine,
            }
            if self.echo_type == "echo1+echo2":
                output["metadata"]["echo"] = example["echo"]
        return output

    def __getitem__(self, i: int):
        """Returns the i-th example in dataset.

        Returns:
            Dict[str, Any]: A dictionary representing the example. Potential keys include:
                * ``"kspace"``: The complex-valued undersampled k-space.
                * ``"maps"``: The complex-valued sensitivity maps.
                * ``"target"``: The complex-valued target (ground-truth) reconstruction image.
                * ``"mask"``: The undersampling mask.
                * ``"metadata"``: A dictionary of metadata (if ``self.include_metadata`` is True):
                    * ``"slice_id"`` (int): The zero-indexed slice id.
                    * ``"scan_id"`` (str): The scan id.
                    * ``"orientation"`` (Tuple[str]): The orientation of volume.
                      The orientation (O1, O2, O3) is ordered such that
                      O1 - the slice dimension, O2 - the height/row dimension,
                      O3 - the width/column dimension.
                    * ``"voxel_spacing"`` (Tuple[float]): The voxel spacing of the volume.
                      The ordering corresponds to the orientation.
                    * ``"affine"`` (np.ndarray): The affine orientation-spacing matrix.
                      See :class:`dosma.MedicalVolume` for more information.
                    * ``"echo"`` (int): The zero-indexed echo number.
                      This is only available when ``self.echo_type == "echo1+echo2"``.
        """
        # Copy so downstream loading/transforms can do whatever they want.
        example = deepcopy(self.examples[i])

        data = self._load_data(example, i)
        if self._include_metadata:
            metadata: Dict[str, Any] = data.pop("metadata", {})
            metadata.update({k: example[k] for k in self._METADATA_KEYS if k not in metadata})
            data["metadata"] = metadata
            data["metadata"]["slice_id"] = example["slice_id"]
        return data

    def get_undersampling_seeds(self):
        """Computed seeds for deterministic undersampling for each scan.

        Note:
            This function is only used with validation data.
            Testing data has precomputed masks distributed with
            the SKM-TEA dataset.

        Note:
            This function should be used for generating deterministic
            undersampling masks when simulating fixed undersampled
            training data in unsupervised/semi-supervised learning
            scenarios.
        """
        seeds = []
        for ex in self.examples:
            fname = ex["scan_id"]
            seeds.append(sum(tuple(map(ord, fname))))
        return seeds


class SkmTeaDicomDataset(SkmTeaRawDataset):
    """The dataset to use for the SKM-TEA DICOM Track.

    This dataset handles loading images and labels related to the DICOM Track.
    Images and segmentations are stored in the HDF5 format. Detection annotations
    are read in from corresponding annotation files.

    Most of the functionality is shared with :class:`SkmTeaRawDataset`.

    Args:
        dataset_dicts (List[Dict]): See :class:`SkmTeaRawDataset`.
        transform (Callable): See :class:`SkmTeaRawDataset`.
        split (str): See :class:`SkmTeaRawDataset`.
        include_metadata (bool, optional): See :class:`SkmTeaRawDataset`.
        tasks (Collection[str], optional): Tasks to return data for.
            Unlike :class:`SkmTeaRawDataset`, ``"recon"`` is not supported.
        seg_classes (Sequence[Union[int, Sequence[int]]]): Integer labels
            for segmentation classes to load.
        keys_to_load (Collection[str]): Keys to load from DICOM Track
            HDF5 files.
        return_batches (bool, optional): See :class:`SkmTeaRawDataset`.
        batch_size (int, optional): See :class:`SkmTeaRawDataset`.
        echo_type (str, optional): The input image type. One of
            ``["echo1", "echo2", "rss", "echo1-echo2-mc"]``.
        cache_files (bool, optional): See :class:`SkmTeaRawDataset`.
        orientation (str, optional): The orientation of the scan for
            segmentation. One of ``["axial", "coronal", "sagittal"]``.
        suppress_fluid (bool, optional): If ``True``, suppress fluid regions
            in input image.
        suppress_fat (bool, optional): If ``True``, suppress fat regions in
            input image.
    """

    _ALIASES = ("DicomTrack", "SkmTeaDicomTrack", "SkmTeaDicomTrackDataset", "qDESSImageDataset")

    def __init__(
        self,
        dataset_dicts: List[Dict],
        transform: Callable,
        split: str,
        keys=None,
        include_metadata: bool = False,
        tasks: Sequence[str] = ("sem_seg",),
        seg_classes: Sequence[Union[int, Sequence[int]]] = None,
        keys_to_load=None,
        echo_type: str = "echo1",
        cache_files: bool = False,
        orientation: str = "axial",
        suppress_fluid: bool = False,
        suppress_fat: bool = False,
    ):
        """
        Args:
            dataset_dicts: See :class:`SkmTeaRawDataset`.
            transform: See :class:`SkmTeaRawDataset`.
            split: See :class:`SkmTeaRawDataset`.
            keys: See :class:`SkmTeaRawDataset`.
            include_metadata: See :class:`SkmTeaRawDataset`.
            tasks: The task(s) to return data for. This dataset currently supports tasks:
                * ``'sem_seg'``: Semantic segmentation
            seg_classes: See :class:`SkmTeaRawDataset`.
            keys_to_load: See :class:`SkmTeaRawDataset`.
            echo_type: See :class:`SkmTeaRawDataset`.
            cache_files: See :class:`SkmTeaRawDataset`.
            orientation: The orientation along which to slice the volume.
                One of ``["axial", "coronal", "sagittal"]``.
            suppress_fluid: Whether to suppress fluid regions in the image during preprocessing.
            suppress_fat: Whether to suppress fat regions in the image during preprocessing.
        """

        if "recon" in tasks:
            raise ValueError("Task 'recon' is not supported on the SKM-TEA Dicom Track.")
        self.orientation = orientation

        # Preprocessing
        self.suppress_fluid = suppress_fluid
        self.suppress_fat = suppress_fat
        if keys_to_load is None:
            keys_to_load = {"seg"}
            if echo_type in ("echo1", "echo2", "rms"):
                keys_to_load |= {echo_type}
            elif echo_type in ("rss", "echo1-echo2-mc"):
                keys_to_load |= {"echo1", "echo2"}
        if self.suppress_fluid:
            keys_to_load |= {"echo1", "echo2"}
        if self.suppress_fat:
            keys_to_load |= {"echo1"}

        super().__init__(
            dataset_dicts=dataset_dicts,
            transform=transform,
            split=split,
            keys=keys,
            include_metadata=include_metadata,
            tasks=tasks,
            seg_classes=seg_classes,
            keys_to_load=keys_to_load,
            echo_type="echo1-echo2-mc" if echo_type in ("rms", "rss") else echo_type,
            cache_files=cache_files,
        )
        self.echo_type = echo_type

    def _init_examples(self, dataset_dicts: List[Dict[str, Any]], slice_dim: Optional[int] = None):
        if slice_dim is None:
            slice_dim = {"axial": 0, "coronal": 1, "sagittal": 2}[self.orientation]
        examples = super()._init_examples(dataset_dicts, slice_dim)
        for ex in examples:
            ex["slice_dim"] = slice_dim
            ex["inplane_shape"] = tuple(
                s for idx, s in enumerate(ex["matrix_shape"]) if idx != slice_dim
            )
            # Permute voxel spacing so that first value corresponds to
            # spacing along the the slice dimension.
            if "voxel_spacing" in ex:
                voxel_spacing = ex["voxel_spacing"]
                ex["voxel_spacing"] = (voxel_spacing[slice_dim],) + tuple(
                    s for idx, s in enumerate(voxel_spacing) if idx != slice_dim
                )
            # Same with orientation
            if "orientation" in ex:
                orientation = ex["orientation"]
                ex["orientation"] = (orientation[slice_dim],) + tuple(
                    o for idx, o in enumerate(orientation) if idx != slice_dim
                )
        return examples

    def _load_files(self):
        # Only load DICOM image data.
        return super()._load_files(file_keys=["image_file"])

    def _build_stats(self, dataset_dicts: List[Dict[str, Any]]):
        files = {dd["image_file"] for dd in dataset_dicts}
        stats = {}
        for fp in files:
            stats[fp] = {}
            with h5py.File(fp, "r") as f:
                for k in ["echo1", "echo2", "rms", "rss"]:
                    if k in f["stats"]:
                        stats[fp][k] = {x: f["stats"][k][x][()] for x in f["stats"][k].keys()}
        return stats

    def preprocess(
        self, example: Dict[str, Any], inputs: Dict[str, np.ndarray], file_key: str = "file_name"
    ):
        # Pre-processing
        fp = example[file_key]
        stats = self.stats[fp]
        img_key = self.echo_type

        # If rss is specified, we probably need to compute it on the fly
        # from echo1 and echo2.
        if img_key == "rss" and "rss" not in inputs:
            inputs["rss"] = np.sqrt(
                inputs["echo1"].astype(np.float32) ** 2 + inputs["echo2"].astype(np.float32) ** 2
            )

        # 1. Zero mean, unit standard deviation wrt volume.
        if img_key == "echo1-echo2-mc":
            img_key = ("echo1", "echo2")
        else:
            img_key = (img_key,)
        image = np.stack(
            [(inputs[k] - stats[k]["mean"]) / stats[k]["std"] for k in img_key], axis=0
        )

        # 2. Suppress fat/fluid regions (optional)
        if self.suppress_fat:
            echo1 = inputs["echo1"][np.newaxis, ...]
            image = image * (echo1 > 0.15 * np.max(echo1))
        if self.suppress_fluid:
            beta = 1.2
            echo1, echo2 = inputs["echo1"][np.newaxis, ...], inputs["echo2"][np.newaxis, ...]
            vol_null_fluid = echo1 - beta * echo2
            image = image * (vol_null_fluid > 0.1 * np.max(vol_null_fluid))

        # 3. Collect/collapse segmentation masks
        sem_seg = inputs["seg"]
        sem_seg = collect_mask(sem_seg, self.seg_idxs, out_channel_first=True)

        return {"image": image.astype(np.float32), "sem_seg": sem_seg.astype(np.float32)}

    @profiler.time_profile()
    def _load_data(self, example, idx):
        """Loads matrix data into the example."""
        slice_id = example["slice_id"]
        slice_dim = example["slice_dim"]
        # scan_id = example["scan_id"]
        # img_key = self.echo_type

        if not hasattr(self, "file_manager"):
            self._load_files()

        # Construct slice.
        sl = [slice(None), slice(None), slice(None)]
        sl[slice_dim] = slice_id
        sl = tuple(sl)

        # Determine orientation
        orientation = example["orientation"]
        voxel_spacing = example["voxel_spacing"]
        affine = dm.to_affine(orientation, spacing=voxel_spacing)

        # Load recon info.
        # data = self.files[example.pop("recon_file")]
        # output["target"] = data[self.mapping["target"]][slice_id, ..., example["echo"], :]
        filepath = example.get("image_file")
        with self.file_manager.yield_file(filepath) as data:
            inputs = {k: data[k][sl] for k in self.keys_to_load}
        output = self.preprocess(example, inputs, file_key="image_file")

        # Do transformation before annotations are converted to 2D instances.
        # output = self.transform(output, scan_id=example['scan_id'], slice_id=slice_id)
        # output["stats"] = {
        #     "target": {
        #         "vol_mean": self.stats["target"][scan_id]["mean"],
        #         "vol_std": self.stats["target"][scan_id]["std"]
        #     }
        # }

        # TODO: Fix this to support sagittal slices.
        if self.use_detection and "annotations" in example:
            img_size = example["matrix_shape"][1:]  # noqa: F841
            # output["instances"] = annotations_to_instances2d(example.pop("annotations"), img_size)
        if self._include_metadata:
            output["metadata"] = {
                "orientation": orientation,
                "voxel_spacing": voxel_spacing,
                "affine": affine,
            }
        return output
