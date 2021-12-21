import pytorch_lightning as pl
from meddlr.data.build import _build_dataset, get_recon_dataset_dicts
from meddlr.data.samplers.build import build_train_sampler, build_val_sampler
from meddlr.data.transforms.subsample import build_mask_func
from torch.utils.data import DataLoader

from skm_tea.data.collate import default_collate
from skm_tea.data.dataset import SkmTeaDicomDataset, SkmTeaRawDataset
from skm_tea.data.transform import qDESSDataTransform

__all__ = ["SkmTeaDataModule"]


_TRACKS_TO_DATASETS = {"raw_data": SkmTeaRawDataset, "dicom": SkmTeaDicomDataset}
_PRECOMPUTED_MASKS_CACHE_DIR = "cache://skm-tea/precomputed-masks"


class SkmTeaDataModule(pl.LightningDataModule):
    """
    TODO:
        - Add support for downloading dataset.
    """

    def __init__(self, cfg, tasks, distributed: bool = False, track=None):
        super().__init__()
        self.cfg = cfg
        self.tasks = tasks
        # self.use_ddp = use_ddp
        self.seg_classes = self.cfg.MODEL.SEG.CLASSES
        self.pin_memory = True
        self.distributed = distributed

        if track is None:
            track = self._get_track_from_cfg()
        if track not in _TRACKS_TO_DATASETS:
            raise ValueError(
                f"`track` must be one of {tuple(_TRACKS_TO_DATASETS.keys())}. Got {track}."
            )
        self.track = track

        # qDESS Dataset specifics.
        self.calib_size = self.cfg.AUG_TRAIN.UNDERSAMPLE.CALIBRATION_SIZE
        acceleration = self.cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS
        if len(acceleration) != 1:
            raise ValueError(
                f"{type(self).__name__} does not support multiple accelerations. Got {acceleration}"
            )
        self.acceleration = float(acceleration[0])
        self.precompute_masks = cfg.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.NUM > 0

    def dataset_type(self, dataset_name):
        return _TRACKS_TO_DATASETS[self.track]

    def prepare_data(self):
        """Data should already be downloaded and processed."""
        # We only do this to build the precomputed masks.
        if self.precompute_masks:
            self._build_train_dataset(self.cfg)

    def setup(self, stage=None):
        cfg = self.cfg

        # Train dataset.
        self.train_dataset = self._build_train_dataset(cfg)

        # Validation/test datasets are not currently configured for filtering.
        self.validation_datasets = self._make_eval_datasets(cfg.DATASETS.VAL, split="val")
        self.test_datasets = self._make_eval_datasets(cfg.DATASETS.TEST, split="test")

    def train_dataloader(self, cfg=None, use_ddp=False):
        if cfg is None:
            cfg = self.cfg
            dataset = self._build_train_dataset(cfg)
        else:
            dataset = self.train_dataset

        # Build sampler.
        sampler, is_batch_sampler = build_train_sampler(cfg, dataset, distributed=use_ddp)
        shuffle = not sampler  # shuffling should be handled by sampler, if specified.
        if is_batch_sampler:
            dl_kwargs = {"batch_sampler": sampler}
        else:
            dl_kwargs = {
                "sampler": sampler,
                "batch_size": cfg.SOLVER.TRAIN_BATCH_SIZE,
                "shuffle": shuffle,
                "drop_last": cfg.DATALOADER.DROP_LAST,
            }

        return DataLoader(
            dataset=dataset,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.pin_memory,
            collate_fn=default_collate,
            prefetch_factor=cfg.DATALOADER.PREFETCH_FACTOR,
            **dl_kwargs,
        )

    def val_dataloader(self, use_ddp=False):
        cfg = self.cfg
        return self._build_eval_dataloaders(self.validation_datasets, cfg, use_ddp)

    def test_dataloader(self, use_ddp=False):
        cfg = self.cfg
        return self._build_eval_dataloaders(self.test_datasets, cfg, use_ddp)

    def _get_track_from_cfg(self):
        track_aliases = {
            track: [dataset.__name__] + list(dataset._ALIASES)
            for track, dataset in _TRACKS_TO_DATASETS.items()
        }
        dataset_type = self.cfg.DATASETS.QDESS.DATASET_TYPE

        # Legacy
        if not dataset_type:
            dataset_type = "raw_data"

        if dataset_type in track_aliases.keys():
            return dataset_type
        for track, aliases in track_aliases.items():
            if dataset_type in aliases:
                return track

        raise ValueError(
            f"Could not determine SKM-TEA challenge track from dataset_type={dataset_type}"
        )

    def _make_eval_datasets(self, dataset_names, split="val"):
        datasets = []
        for dataset_name in dataset_names:
            dataset_dicts = get_recon_dataset_dicts(
                dataset_names=[dataset_name],
                filter_by=None,
                num_scans_total=self.cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_VAL,
            )

            aug_cfg = self.cfg.AUG_TRAIN.clone()
            if split == "test":
                aug_cfg.defrost()
                aug_cfg.UNDERSAMPLE.ACCELERATIONS = self.cfg.AUG_TEST.UNDERSAMPLE.ACCELERATIONS
                aug_cfg.freeze()
                assert len(aug_cfg.UNDERSAMPLE.ACCELERATIONS) == 1
            # Use sigpy backend for Poisson Disc generation.
            # Related to meddlr issue #3.
            mask_func_kwargs = (
                {"module": "sigpy"} if aug_cfg.UNDERSAMPLE.NAME == "PoissonDiskMaskFunc" else {}
            )
            mask_func = build_mask_func(aug_cfg, **mask_func_kwargs)
            data_transform = qDESSDataTransform(self.cfg, mask_func=mask_func, is_test=True)
            dataset_kwargs = {
                k: v
                for k, v in zip(
                    self.cfg.DATASETS.QDESS.KWARGS[::2], self.cfg.DATASETS.QDESS.KWARGS[1::2]
                )
            }
            dataset = _build_dataset(
                self.cfg,
                dataset_dicts,
                data_transform,
                self.dataset_type(dataset_name),
                split=split,
                tasks=self.tasks,
                seg_classes=self.seg_classes,
                is_eval=True,
                echo_type=self.cfg.DATASETS.QDESS.ECHO_KIND,
                **dataset_kwargs,
            )
            if "recon" in self.tasks and split != "test":
                data_transform._subsampler.precompute_masks(
                    acq_shapes={x["acq_shape"] for x in dataset.examples},
                    seed=list(set(dataset.get_undersampling_seeds())),
                    num_workers=self.cfg.DATALOADER.NUM_WORKERS
                    if aug_cfg.UNDERSAMPLE.PRECOMPUTE.USE_MULTIPROCESSING
                    else 0,
                )
            datasets.append(dataset)
        return datasets

    def _build_train_dataset(self, cfg):
        # Train dataset.
        dataset_dicts = get_recon_dataset_dicts(
            dataset_names=cfg.DATASETS.TRAIN,
            num_scans_total=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_TOTAL,
            num_scans_subsample=cfg.DATALOADER.SUBSAMPLE_TRAIN.NUM_UNDERSAMPLED,
            seed=cfg.DATALOADER.SUBSAMPLE_TRAIN.SEED,
            accelerations=cfg.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS,
            filter_by=cfg.DATALOADER.FILTER.BY,
        )
        # Use sigpy backend for Poisson Disc generation.
        # Related to meddlr issue #3.
        mask_func_kwargs = (
            {"module": "sigpy"} if cfg.AUG_TRAIN.UNDERSAMPLE.NAME == "PoissonDiskMaskFunc" else {}
        )
        mask_func = build_mask_func(cfg.AUG_TRAIN, **mask_func_kwargs)
        data_transform = qDESSDataTransform(cfg, mask_func=mask_func, is_test=False)
        dataset_kwargs = {
            k: v for k, v in zip(cfg.DATASETS.QDESS.KWARGS[::2], cfg.DATASETS.QDESS.KWARGS[1::2])
        }
        dataset = _build_dataset(
            cfg,
            dataset_dicts,
            data_transform,
            self.dataset_type(cfg.DATASETS.TRAIN[0]),
            split="train",
            tasks=self.tasks,
            seg_classes=self.seg_classes,
            echo_type=cfg.DATASETS.QDESS.ECHO_KIND,
            **dataset_kwargs,
        )
        if self.precompute_masks:
            data_transform._subsampler.precompute_masks(
                acq_shapes={x["acq_shape"] for x in dataset.examples},
                N=cfg.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.NUM,
                seed=cfg.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.SEED,
                cache=_PRECOMPUTED_MASKS_CACHE_DIR,
                num_workers=self.cfg.DATALOADER.NUM_WORKERS
                if cfg.AUG_TRAIN.UNDERSAMPLE.PRECOMPUTE.USE_MULTIPROCESSING
                else 0,
            )
        return dataset

    def _build_eval_dataloaders(self, datasets, cfg, use_ddp):
        dataloaders = []
        for dataset in datasets:
            sampler, is_batch_sampler = build_val_sampler(
                cfg, dataset, distributed=use_ddp, dist_group_by="scan_id"
            )
            if is_batch_sampler:
                dl_kwargs = {"batch_sampler": sampler}
            else:
                dl_kwargs = {
                    "sampler": sampler,
                    "batch_size": cfg.SOLVER.TEST_BATCH_SIZE,
                    "shuffle": False,
                    "drop_last": False,
                }

            dataloaders.append(
                DataLoader(
                    dataset=dataset,
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    pin_memory=self.pin_memory,
                    collate_fn=default_collate,
                    **dl_kwargs,
                )
            )
        return dataloaders
