"""General utilities for the SKM-TEA dataset."""
import json
import logging
import os
import re
import time
import warnings
from types import SimpleNamespace
from typing import Any, Dict, Sequence

import h5py
import pandas as pd
from meddlr.data.catalog import DatasetCatalog, MetadataCatalog

from skm_tea.utils import env

__all__ = []

logger = logging.getLogger(__name__)
_path_mgr = env.get_path_manager()

# ==============================================
# Paths
# ==============================================
_PATHS = {"v1.0.0": "data://skm-tea/v1-release"}
_VERSION_ALIASES = {"v1.0.0": ["v1"]}

# ==============================================
# Metadata
# ==============================================

SKMTEA_DETECTION_CATEGORIES = [
    {
        "color": [220, 20, 60],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 1,
        "name": "Meniscal Tear (Myxoid)",
    },
    {
        "color": [119, 11, 32],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 2,
        "name": "Meniscal Tear (Horizontal)",
    },
    {
        "color": [0, 0, 142],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 3,
        "name": "Meniscal Tear (Radial)",
    },
    {
        "color": [0, 0, 230],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 4,
        "name": "Meniscal Tear (Vertical/Longitudinal)",
    },
    {
        "color": [106, 0, 228],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 5,
        "name": "Meniscal Tear (Oblique)",
    },
    {
        "color": [0, 60, 100],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 6,
        "name": "Meniscal Tear (Complex)",
    },
    {
        "color": [0, 80, 100],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 7,
        "name": "Meniscal Tear (Flap)",
    },
    {
        "color": [0, 0, 70],
        "supercategory": "Meniscal Tear",
        "supercategory_id": 1,
        "id": 8,
        "name": "Meniscal Tear (Extrusion)",
    },
    {
        "color": [0, 0, 192],
        "supercategory": "Ligament Tear",
        "supercategory_id": 2,
        "id": 9,
        "name": "Ligament Tear (Low-Grade Sprain)",
    },
    {
        "color": [250, 170, 30],
        "supercategory": "Ligament Tear",
        "supercategory_id": 2,
        "id": 10,
        "name": "Ligament Tear (Moderate Grade Sprain or Mucoid Degeneration)",
    },
    {
        "color": [100, 170, 30],
        "supercategory": "Ligament Tear",
        "supercategory_id": 2,
        "id": 11,
        "name": "Ligament Tear (Full Thickness/Complete Tear)",
    },
    {
        "color": [220, 220, 0],
        "supercategory": "Cartilage Lesion",
        "supercategory_id": 3,
        "id": 12,
        "name": "Cartilage Lesion (1)",
    },
    {
        "color": [175, 116, 175],
        "supercategory": "Cartilage Lesion",
        "supercategory_id": 3,
        "id": 13,
        "name": "Cartilage Lesion (2A)",
    },
    {
        "color": [250, 0, 30],
        "supercategory": "Cartilage Lesion",
        "supercategory_id": 3,
        "id": 14,
        "name": "Cartilage Lesion (2B)",
    },
    {
        "color": [0, 226, 252],
        "supercategory": "Cartilage Lesion",
        "supercategory_id": 3,
        "id": 15,
        "name": "Cartilage Lesion (3)",
    },
    {
        "color": [182, 182, 255],
        "supercategory": "Effusion",
        "supercategory_id": 4,
        "id": 16,
        "name": "Effusion",
    },
]

SKMTEA_SEGMENTATION_CATEGORIES = [
    {"color": [64, 170, 64], "id": 0, "name": "Patellar Cartilage", "abbrev": "pc"},
    {"color": [152, 251, 152], "id": 1, "name": "Femoral Cartilage", "abbrev": "fc"},
    {"color": [208, 229, 228], "id": 2, "name": "Tibial Cartilage (Medial)", "abbrev": "tc-m"},
    {"color": [206, 186, 171], "id": 3, "name": "Tibial Cartilage (Lateral)", "abbrev": "tc-l"},
    {"color": [152, 161, 64], "id": 4, "name": "Meniscus Cartilage (Medial)", "abbrev": "men-m"},
    {"color": [116, 112, 0], "id": 5, "name": "Meniscus Cartilage (Lateral)", "abbrev": "men-l"},
    {"color": [0, 114, 143], "id": (2, 3), "name": "Tibial Cartilage", "abbrev": "tc"},
    {"color": [102, 102, 156], "id": (4, 5), "name": "Meniscus", "abbrev": "men"},
]


# ==============================================
# Helper functions.
# ==============================================


def get_paths(version):
    if version in _PATHS:
        data_dir = _PATHS[version]
    else:
        aliases = {alias: base for base, aliases in _VERSION_ALIASES.items() for alias in aliases}
        data_dir = _PATHS[aliases[version]]
    return SimpleNamespace(
        metadata_csv=f"{data_dir}/all_metadata.csv",
        mask_gradwarp_corrected=f"{data_dir}/segmentation_masks/raw-data-track",
        image_files=f"{data_dir}/image_files",
        recon_files=f"{data_dir}/files_recon_calib-24",
        dicom_files=f"{data_dir}/dicoms",
        dicom_masks=f"{data_dir}/segmentation_masks/dicom-track",
        ann_dir=f"{data_dir}/annotations/{version}",
    )


def _get_version_from_name(name):
    version = re.findall("v[0-9]+[0-9|\.]*", name)
    assert len(version) == 1, version
    return version[0]


# TODO: probably best to pass a json file here to make sure we can keep up with
# the ever changing annotation files. Segmentations should stay constant though.
def get_skmtea_instances_meta(version, group_instances_by=None) -> Dict[str, Any]:
    """

    Args:
        group_by (str, optional): How to group detection labels.
            Currently only supports grouping by "supercategory".
    """
    assert group_instances_by in [None, "supercategory"], f"group_by={group_instances_by}"

    path_manager = env.get_path_manager()

    if group_instances_by is None:
        thing_ids = [k["id"] for k in SKMTEA_DETECTION_CATEGORIES]
        thing_classes = [k["name"] for k in SKMTEA_DETECTION_CATEGORIES]
        thing_colors = [k["color"] for k in SKMTEA_DETECTION_CATEGORIES]
    elif group_instances_by == "supercategory":
        things = {
            k["supercategory_id"]: (k["supercategory"], k["color"])
            for k in SKMTEA_DETECTION_CATEGORIES
        }
        thing_ids = list(things.keys())
        thing_classes = [v[0] for v in things.values()]
        thing_colors = [v[1] for v in things.values()]
    else:
        raise ValueError(f"{group_instances_by} not supported")

    # Mapping from the incontiguous qDESS category id to an id in [0, N]
    # N=15 generally, N=4 if group by supercategory
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}

    # Segmentation classes
    # TODO: Add support for subselecting classes.
    # seg_classes = [k["name"] for k in QDESS_SEGMENTATION_CATEGORIES]
    # seg_colors = [k["color"] for k in QDESS_SEGMENTATION_CATEGORIES]
    # seg_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    # seg_abbrevs = [k["abbrev"] for k in QDESS_SEGMENTATION_CATEGORIES]

    paths = get_paths(version)

    ret = {
        # Detection
        "group_instances_by": group_instances_by,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
        "scan_metadata": pd.read_csv(path_manager.get_local_path(paths.metadata_csv), index_col=0),
        # This mask path is temporary. In the future, the segmentations will be made
        # available directly through the recon h5 file.
        "mask_gw_corr_dir": path_manager.get_local_path(paths.mask_gradwarp_corrected),
        "version": version,
    }
    return ret


def load_skmtea_annotations(
    json_file: str,
    dataset_name: str,
    recon_root: str = None,
    image_root: str = None,
    calib_size: int = None,
):
    """Load SKM-TEA json annotation files.

    Metadata must be initialized for ``dataset_name``.

    Args:
        json_file (str): Path to json file containing annotations.
        recon_root (str): Path to directory with recon data.
        image_root (str): Path to directory with image files (images, segmentations, etc).
        dataset_name (str): Name of the dataset.

    Returns:
        List[Dict]: List of dictionaries corresponding to scan data.
    """
    version = _get_version_from_name(dataset_name)

    paths = get_paths(version)
    if recon_root is None:
        recon_root = paths.recon_files
    if image_root is None:
        image_root = paths.image_files

    # In some cases, we want to populate the recon_root dynamically.
    # In other cases, recon_root will already be a full filepath without
    # need for formatting. In the latter case, recon_root is unchanged
    # by the syntax below.
    recon_root = recon_root.format(calib_size=calib_size)

    json_file = _path_mgr.get_local_path(json_file)
    start_time = time.perf_counter()
    with open(json_file, "r") as f:
        data = json.load(f)
    logger.info(
        "Loading {} takes {:.2f} seconds".format(json_file, time.perf_counter() - start_time)
    )

    # meta = MetadataCatalog.get(dataset_name)
    # group_instances_by = meta.group_instances_by

    # TODO: Add any relevant metadata.
    start_time = time.perf_counter()
    dataset_dicts = []
    for d in data["images"]:
        dd = dict(d)

        if "sagittal-ds" in dataset_name:
            dd["matrix_shape"][2] = int(dd["matrix_shape"][2] / 2)
            dd["voxel_spacing"][2] *= 2

        # Recon File
        file_name = _path_mgr.get_local_path(os.path.join(recon_root, d["file_name"]))
        dd["recon_file"] = file_name

        # Image file
        msp_file_name = _path_mgr.get_local_path(os.path.join(image_root, d["file_name"]))
        dd["image_file"] = msp_file_name

        # Gradient-warp corrected mask nifti file.
        dd["gw_corr_mask_file"] = _path_mgr.get_local_path(
            os.path.join(paths.mask_gradwarp_corrected, f"{d['scan_id']}.nii.gz")
        )

        # Dicom paths
        dd["dicom_mask_file"] = _path_mgr.get_local_path(
            os.path.join(paths.dicom_masks, f"{d['scan_id']}.nii.gz")
        )
        dd["dicom_dir"] = _path_mgr.get_local_path(os.path.join(paths.dicom_files, d["scan_id"]))

        # Drop keys that are not needed.
        for k in ["msp_id", "msp_file_name"]:
            dd.pop(k, None)

        # Load number of coils for filtering purposes.
        with h5py.File(dd["recon_file"], "r") as f:
            dd["num_coils"] = f["kspace"].shape[-1]

        # TODO: Add support for annotations.

        dataset_dicts.append(dd)

    logger.info(
        "Formatting dataset dicts takes {:.2f} seconds".format(time.perf_counter() - start_time)
    )

    return dataset_dicts


def seg_categories_to_idxs(categories: Sequence[str]):
    """Converts segmentation names/abbreviations to relevant indices.

    Args:
        categories (Sequence[str]): Names or abbreviations for segmentation classes.

    Returns:
        Sequence[Union[int, Tuple[int]]]: Ordered indices or group of indices for
            segmentation classes.
    """
    mappings = {k["abbrev"].lower(): k["id"] for k in SKMTEA_SEGMENTATION_CATEGORIES}
    mappings.update({k["name"].lower(): k["id"] for k in SKMTEA_SEGMENTATION_CATEGORIES})
    idxs = []
    for cat in categories:
        if isinstance(cat, str):
            cat = cat.lower()
            try:
                idx = mappings[cat]
            except KeyError:
                raise ValueError("category {} unknown for qDESS dataset".format(cat))
        elif isinstance(cat, int) or all(isinstance(x, int) for x in cat):
            idx = cat
        else:
            raise ValueError("category {} unknown for qDESS dataset".format(cat))
        idxs.append(idx)
    return idxs


def _build_predefined_splits():
    splits = {}
    for version in _PATHS:
        paths = get_paths(version)
        for split in ["train", "val", "test"]:
            name = f"skmtea_{version}_{split}"
            splits[name] = (None, os.path.join(paths.ann_dir, f"{split}.json"))
            for v_alias in _VERSION_ALIASES.get(version, []):
                splits[f"skmtea_{v_alias}_{split}"] = (
                    None,
                    os.path.join(paths.ann_dir, f"{split}.json"),
                )
    return splits


def register_skm_tea(name, json_file, metadata: Dict[str, Any] = None):
    # 1. register a function which returns dicts
    version = _get_version_from_name(name)
    paths = get_paths(version)
    recon_dir = _path_mgr.get_local_path(paths.recon_files)
    image_dir = _path_mgr.get_local_path(paths.image_files)
    mask_gradwarp_corrected_dir = _path_mgr.get_local_path(paths.mask_gradwarp_corrected)
    DatasetCatalog.register(
        name,
        lambda calib_size=None: load_skmtea_annotations(json_file, name, calib_size=calib_size),
    )

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    base_metadata = get_skmtea_instances_meta(version=version, group_instances_by=None)
    if metadata is not None:
        base_metadata.update(metadata)
    metadata = base_metadata
    MetadataCatalog.get(name).set(
        json_file=json_file,
        recon_dir=recon_dir,
        image_dir=image_dir,
        raw_data_track_dir=recon_dir,
        dicom_track_dir=image_dir,
        mask_gradwarp_corrected_dir=mask_gradwarp_corrected_dir,
        evaluator_type="SkmTeaEvaluator",
        **metadata,
    )


def register_all_skm_tea():
    for name, init_args in _build_predefined_splits().items():
        _, ann_file = init_args
        register_skm_tea(name, json_file=ann_file)


try:
    register_all_skm_tea()
except Exception as e:
    warnings.warn(
        "SKM-TEA dataset was not properly registered. "
        "Please check that the dataset was properly downloaded and paths have been set. "
        "The error is shown below:\n{}".format(e)
    )
