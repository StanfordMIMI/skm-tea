import itertools
import logging
import os
from functools import partial
from typing import Any, Callable, Collection, Dict, Tuple, Union

import meddlr.ops as oF
import numpy as np
import torch
import zarr
from meddlr.data.transforms.transform import AffineNormalizer, build_normalizer
from meddlr.forward import SenseModel
from meddlr.ops import complex as cplx
from meddlr.utils import profiler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from skm_tea.modeling.build import get_model_cfg
from skm_tea.utils import env

__all__ = ["CachingSubsampler", "qDESSDataTransform"]
_logger = logging.getLogger(__name__)


class CachingSubsampler:
    """A subsampler that supports precomputing and caching masks."""

    def __init__(self, mask_func):
        self.mask_func = mask_func

        # TODO: Determine what the memory bandwidth of this is.
        self._precomputed_masks = None
        self._seeds: Dict[int, int] = None

        # Fields that should be set by individual workers:
        # TODO: Make this reproducible and different among each worker.
        seed = os.environ.get("PYTHONHASHSEED", None)
        if seed is not None:
            seed = int(seed)
        self._rng = np.random.RandomState(seed=seed)

    def precompute_masks(
        self,
        acq_shapes: Collection[Tuple[int]],
        N: int = None,
        accelerations: Collection[float] = None,
        seed=None,
        mode="2D",
        cache: Union[bool, str] = False,
        read_only=True,
        num_workers: int = 0,
    ):
        """Precomputes masks and keeps in memory.

        Args:
            acq_shapes (Collection[Tuple[int]]): The acquisition shapes to generate masks for.
            N (int, optional): Number of masks to generate. Defaults to ``len(seed)``.
                Must be specified if ``seeds`` is not a list of seeds.
            accelerations (Collection[float], optional): Discrete accelerations for generating
                masks. If not specified, defaults to choosing acceleration via ``self.mask_func``.
                This must be specified when ``is_eval=True``.
            seed (int | Sequence[int], optional): Seeds to use to generate masks.
                If an integer, the seed is used to initialize another random number generator
                which produces the seeds to ensure reproducibility. Defaults to no seed.
            mode (str): The undersampling mode.
            is_eval (bool, optional): If ``True``, generate masks as if in eval mode.
            cache (bool | str, optional): If ``True``, masks are cached for use in future runs.
                Note this is not thread safe. To ensure appropriate caching for masks, only cache
                with one process. If ``str``, this corresponds to the directory where to cache
                files. Caching is not supported when ``seed`` is a list.
            read_only (bool, optional): If ``True`` and cache file exists, modifications are not
                made directly to the cache file. If ``False``, fields that need to be updated are
                updated in the cache file.
        """
        rand_state = None
        cache_dir = cache if isinstance(cache, str) else "cache://"
        cache_dir = env.get_path_manager().get_local_path(cache_dir)

        if not N and not isinstance(seed, (Collection, np.ndarray)):
            raise ValueError("Either `N` or collection of seeds needs to be specified")

        if cache and isinstance(seed, (Collection, np.ndarray)):
            raise ValueError("Caching is not supported for multiple seeds")

        if isinstance(seed, int) and seed < 0:
            seed = None

        base_seed = seed
        if N:
            assert isinstance(seed, (int, type(None)))
            if seed is None:
                _logger.warning(
                    "Seed not specified when generating masks. "
                    "This can result in non-reproducible behavior"
                )
            else:
                rand_state = np.random.get_state()
                np.random.seed(seed)
                seed = 10000000 + np.random.choice(10000000, size=N, replace=False)
        else:
            N = len(seed)

        # Create/open the zarr file.
        if cache:
            fname = (
                f"N={N}-acc={accelerations}-mode={mode}-seed={base_seed}--"
                f"{self.mask_func.get_str_name()}.zarr"
            )
            filepath = os.path.join(cache_dir, fname)
            _logger.info(f"Caching masks to {filepath}")
            if os.path.exists(filepath):
                if read_only:
                    _logger.info(f"Found existing cache file - {filepath}")
                    self._precomputed_masks = zarr.open(filepath, mode="r")
                    self._seeds = (
                        {p_seed: idx for idx, p_seed in enumerate(seed)}
                        if seed is not None
                        else None
                    )
                    return
                root = zarr.open(filepath, mode="r" if read_only else "r+")
            else:
                root = zarr.group(filepath)
        else:
            root = {}

        fields = [acq_shapes, accelerations]
        fields = [[x] if not isinstance(x, (Collection, np.ndarray)) else x for x in fields]
        nbytes = 0
        for acq_shape, acc in tqdm(
            list(itertools.product(*fields)), desc="Precomputing masks", leave=True
        ):
            nbytes += N * np.prod(acq_shape)

            key = (acq_shape, acc, mode)
            if key in root:
                continue

            mask_kwargs_base = {"shape": (1,) + acq_shape, "acceleration": acc}
            if seed is None:
                seed = [None] * N
            mask_kwargs = [mask_kwargs_base.copy() for _ in range(len(seed))]
            for kwargs, p_seed in zip(mask_kwargs, seed):
                kwargs.update({"seed": p_seed})

            if num_workers > 0:
                func = partial(_precompute_mask, mask_func=self.mask_func)
                max_workers = min(num_workers, len(mask_kwargs))
                masks = process_map(func, mask_kwargs, max_workers=max_workers)
            else:
                masks = []
                for mkwargs in tqdm(mask_kwargs):
                    masks.append(_precompute_mask(mkwargs, self.mask_func))
            masks = torch.cat(masks, dim=0).type(torch.bool).numpy()

            if isinstance(root, Dict):
                root[key] = masks
            else:
                root.create_dataset(key, data=masks)

        self._precomputed_masks = root
        self._seeds = {p_seed: idx for idx, p_seed in enumerate(seed)} if seed is not None else None

        if rand_state is not None:
            np.random.set_state(rand_state)

    def _get_mask_shape(self, data_shape, mode: str):
        """Returns the shape of the mask based on the data shape.

        Args:
            data_shape (tuple[int]): The data shape.
            mode: Either ``"2D"`` or ``"3D"``
        """
        if mode == "2D":
            extra_dims = len(data_shape) - 3
            mask_shape = (1,) + data_shape[1:3] + (1,) * extra_dims
        elif mode == "3D":
            extra_dims = len(data_shape) - 4
            mask_shape = (1,) + data_shape[1:4] + (1,) * extra_dims
        else:
            raise ValueError("Only 2D and 3D undersampling masks are supported.")
        return mask_shape

    def __call__(
        self,
        data: torch.Tensor,
        mode: str = "2D",
        seed: int = None,
        acceleration: int = None,
        acq_shape: Tuple[int] = None,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            data (torch.Tensor): The batched kspace data.
                Shape ``(N, kx, ky, ...)`` for ``mode=2D``
                or ``(N, kx, ky, kz, ...)`` for ``mode=3D`.
            mode (str, optional): Either ``'2D'`` for 2D undersampling or
                ``'3D'`` for 3D undersampling.
            acceleration (float, optional): The acceleration of the scan to generate.
                Defaults to randomly choosing an acceleration based on ``self.mask_func``.
            acq_shape (Tuple[int], optional): The acquisition matrix size.
                If specified, a mask of this shape will be generated and zero-padded
                to the k-space dimensions of ``data.shape``. Defaults to ``data.shape[1:3]``
                or ``data.shape[1:4]`` for 2D/3D, respectively.
            mask (torch.Tensor): A precomputed mask that is plumbed through.
                If this is provided, the subsampler will use this mask.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The subsampled kspace and the subsampling mask.
        """
        assert mode in ["2D", "3D"]
        data_shape = tuple(data.shape)
        if not acq_shape:
            acq_shape = data_shape[1:3] if mode == "2D" else data_shape[1:4]
        else:
            assert len(acq_shape) in (2, 3)
        zero_pad = any(acq_dim != data_dim for acq_dim, data_dim in zip(acq_shape, data_shape[1:]))

        # Build mask
        extra_dims = len(data_shape) - 1 - len(acq_shape)
        acq_shape_extended = (1,) + acq_shape + (1,) * extra_dims
        mask_shape = self._get_mask_shape(acq_shape_extended, mode)
        if mask is None:
            if self._precomputed_masks is not None:
                all_masks = self._precomputed_masks[(acq_shape, acceleration, mode)]
                if seed is not None:
                    mask = all_masks[self._seeds[seed]]
                else:
                    mask = all_masks[int(self._rng.choice(len(all_masks)))]
                mask = torch.from_numpy(mask.reshape(mask_shape))
            else:
                mask = self.mask_func(mask_shape, seed, acceleration)
        else:
            mask = mask.reshape(mask_shape)

        if zero_pad:
            padded_mask_shape = self._get_mask_shape(data_shape, mode)
            # Batch dimension is not passed to padding.
            mask = oF.zero_pad(mask, padded_mask_shape[1:])
        return torch.where(mask == 0, torch.tensor([0], dtype=data.dtype), data), mask


class _DataTransform:
    """
    Data Transformer for training unrolled reconstruction models.

    For scans that
    """

    def __init__(self, cfg, mask_func, is_test: bool = False, add_noise: bool = False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a
                mask of appropriate shape.
            is_test (bool): If `True`, this class behaves with test-time
                functionality. In particular, it computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
                TODO (#1): Rename to `is_eval`.
        """
        self._cfg = cfg
        self.mask_func = mask_func
        self._is_test = is_test

        try:
            model_cfg = get_model_cfg(cfg)
            self.use_magnitude = model_cfg.get("USE_MAGNITUDE", False)
        except (KeyError, ValueError):
            self.use_magnitude = False

        # Build subsampler.
        # mask_func = build_mask_func(cfg)
        self._subsampler = CachingSubsampler(self.mask_func)
        self.add_noise = add_noise
        seed = cfg.SEED if cfg.SEED > -1 else None
        self.rng = np.random.RandomState(seed)
        self.p_noise = cfg.AUG_TRAIN.NOISE_P
        self._normalizer = build_normalizer(cfg)

        self._normalizer = AffineNormalizer()

    @profiler.time_profile()
    def __call__(
        self,
        kspace,
        maps,
        target,
        fname,
        slice_id,
        is_fixed,
        acceleration: int = None,
        scale: float = None,
        bias: float = None,
        is_batch: bool = False,
        acq_shape: Tuple[int] = None,
        mask: torch.Tensor = None,
    ):
        """
        Args:
            kspace (numpy.array): Input k-space of shape
                (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5
                object.
            fname (str): File name
            slice (int): Serial number of the slice.
            is_fixed (bool, optional): If `True`, transform the example
                to have a fixed mask and acceleration factor.
            acceleration (int): Acceleration factor. Must be provided if
                `is_undersampled=True`.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        # Timer keys
        _GENERATE_AND_APPLY_MASK = "generate_and_apply_mask"
        _ZF_RECON = "zero_filled_recon"
        timer = profiler.get_timer()

        if is_fixed and not acceleration:
            raise ValueError("Accelerations must be specified for undersampled scans")

        # Convert everything from numpy arrays to tensors
        kspace = cplx.to_tensor(kspace).unsqueeze(0)
        maps = cplx.to_tensor(maps).unsqueeze(0)
        target_init = cplx.to_tensor(target).unsqueeze(0)
        target = (
            torch.complex(target_init, torch.zeros_like(target_init)).unsqueeze(-1)
            if not torch.is_complex(target_init)
            else target_init
        )  # handle rss vs. sensitivity-integrated
        norm = torch.sqrt(torch.mean(cplx.abs(target) ** 2))

        # TODO: Add other transforms here.

        # Apply mask in k-space
        seed = sum(tuple(map(ord, fname))) if self._is_test or is_fixed else None  # noqa
        timer.start(_GENERATE_AND_APPLY_MASK)
        masked_kspace, mask = self._subsampler(
            kspace, mode="2D", seed=seed, acceleration=acceleration, acq_shape=acq_shape, mask=mask
        )
        timer.stop(_GENERATE_AND_APPLY_MASK)

        # Zero-filled Sense Recon.
        timer.start(_ZF_RECON)
        if torch.is_complex(target_init):
            A = SenseModel(maps, weights=mask)
            image = A(masked_kspace, adjoint=True)
        # Zero-filled RSS Recon.
        else:
            image = oF.ifft2c(masked_kspace)
            image_rss = torch.sqrt(torch.sum(cplx.abs(image) ** 2, axis=-1))
            image = torch.complex(image_rss, torch.zeros_like(image_rss)).unsqueeze(-1)
        timer.stop(_ZF_RECON)

        # Use magnitude
        if self.use_magnitude:
            image = cplx.abs(image)
            target = cplx.abs(target)

        # Normalize
        normalizer_args = {
            "masked_kspace": masked_kspace,
            "image": image,
            "target": target,
            "mask": mask,
        }
        if isinstance(self._normalizer, AffineNormalizer):
            normalized = self._normalizer.normalize(scale=scale, bias=bias, **normalizer_args)
        else:
            normalized = self._normalizer.normalize(**normalizer_args)

        masked_kspace = normalized["masked_kspace"]
        target = normalized["target"]
        image = normalized["image"]
        mean = torch.as_tensor(normalized["mean"])
        std = torch.as_tensor(normalized["std"])
        # mean = normalized["mean"]
        # std = normalized["std"]

        add_noise = self.add_noise and (
            self._is_test or (not is_fixed and self.rng.uniform() < self.p_noise)
        )
        if add_noise:
            # Seed should be different for each slice of a scan.
            noise_seed = seed + slice_id if seed is not None else None
            masked_kspace = self.noiser(masked_kspace, mask=mask, seed=noise_seed)

        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        maps = maps.squeeze(0)
        target = target.squeeze(0)
        image = image.squeeze(0)

        if not self.use_magnitude:
            image = None

        return masked_kspace, maps, target, mean, std, norm, image


class qDESSDataTransform(_DataTransform):
    """Data transform for SKM-TEA dataset.

    Returns the following keys:
        - "kspace": undersampled preprocessed kspace image
        - "target": preprocessed fully sampled target image
        - "sem_seg": semantic segmentation mask
    """

    def __init__(self, cfg, mask_func, is_test: bool = False):
        """
        Args:
            mask_func (utils.subsample.MaskFunc): A function that can create a
                mask of appropriate shape.
            is_test (bool): If `True`, this class behaves with test-time
                functionality. In particular, it computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
                TODO (#1): Rename to `is_eval`.
        """
        super().__init__(cfg, mask_func, is_test)
        self.tasks = cfg.MODEL.TASKS

    def __call__(
        self,
        example,
        scan_id,
        slice_id=None,
        acceleration: int = None,
        scale=None,
        bias=None,
        acq_shape=None,
    ):
        """Transforms example.

        Note:
            `example` will be modified. If needed, please create a deep copy.

        Args:
            example (Dict[str, Any]): Dictionary representation of the example with keys:
                Required:
                    * "kspace" (ndarray, np.complex64): Input k-space. Shape (H, W, #coils)
                        for multi-coil data or (row)
                    * "maps" (ndarray, np.complex64): Input sensitivity maps.
                        Shape (H, W, # coils, # maps)
                    * "target" (ndarray, np.complex64): Target image. Shape (H, W, # maps)
                Optional:
                    * "annotations" (List[Dict]): Annotations for the given example.
                    * "
            scan_id (str): The scan id.
            slice_id (int, optional): The slice id. Currently does not have any impact on
                performance.
            acceleration (int): Acceleration factor, typically provided if `is_undersampled=True`.
                Note if `self.mask_func` is of type `qDESSLoadMaskFunc`, the acceleration will be
                ignored.

        Returns:
            Dict: The transformed example with the following keys. If last dimension is `2`,
                it corresponds to real/imaginary respectively:
                * "kspace" (torch.Tensor): Preprocessed k-space. Shape (H, W, #coils, 2).
                * "maps" (torch.Tensor): Sensitivity maps. Shape (H, W, #coils, #maps, 2)
                * "target" (torch.Tensor): Preprocessed fully-sampled target. Shape (H, W, #maps, 2)
                * "mean" (torch.Tensor): Normalization mean.
                * "std" (torch.Tensor): Normalization standard deviation
        """
        if "recon" in self.tasks:
            masked_kspace, maps, target, mean, std, norm, image = super().__call__(
                example["kspace"],
                example["maps"],
                example["target"],
                fname=scan_id,
                slice_id=slice_id,
                is_fixed=False,
                acceleration=acceleration,
                scale=scale,
                bias=bias,
                acq_shape=acq_shape,
                mask=example.pop("mask", None),
            )
        else:
            # Normalization done by the model.
            masked_kspace, maps, target, mean, std, norm, image = (
                None,
                None,
                example["target"],
                torch.as_tensor([0.0]),
                torch.as_tensor([1.0]),
                torch.as_tensor([1.0]),
                None,
            )

        for key, data in zip(
            ["kspace", "maps", "target", "mean", "std", "norm", "zf_image"],
            [masked_kspace, maps, target, mean, std, norm, image],
        ):
            if data is None:
                continue
            example[key] = data

        # TODO: Add any transformation of segmentations, bounding boxes, etc.
        return example


def _precompute_mask(mask_kwargs: Dict[str, Any], mask_func: Callable = None):
    if mask_kwargs is None:
        mask_kwargs = {}
    mask_kwargs = mask_kwargs.copy()

    seed = mask_kwargs.get("seed", None)
    shape = mask_kwargs.pop("shape")

    if seed is None:
        out = None
        while out is None:
            try:
                out = mask_func(shape, **mask_kwargs)
            except ValueError:
                continue
        return out
    else:
        return mask_func(shape, **mask_kwargs)
