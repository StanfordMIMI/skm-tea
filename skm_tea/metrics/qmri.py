import inspect
import logging
import os
import warnings
from typing import Dict, Sequence, Union

import dosma as dm
import numpy as np
import pandas as pd
import torch
from dosma.core.orientation import SAGITTAL
from dosma.core.quant_vals import T2
from dosma.tissues import FemoralCartilage, Meniscus, PatellarCartilage, TibialCartilage
from meddlr.metrics import Metric
from torchmetrics.utilities import reduce
from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ["QuantitativeKneeMRI"]


class QuantitativeKneeMRI(Metric):
    """Metric for computing quantitative MRI parameters for knee tissues.

    If ``use_subregions=True``, tissues that are supported for subdivision by DOSMA
    are subdivided into clinically relevant regions. As of v0.1.0, DOSMA currently
    supports subdivision for femoral cartilage, tibial cartilage, and patellar cartilage.

    See :cls:`meddlr.metrics.Metric` for argument details.

    Note:
        This metric does not preserve gradients. It should not be used for loss computation.
    """

    _SUPPORTED_TISSUES = {"fc", "tc", "pc", "men"}
    is_differentiable = False

    def __init__(
        self,
        subregions: bool = False,
        channel_names: Sequence[str] = None,
        units: str = None,
        reduction="none",
        compute_on_step: bool = False,
        dist_sync_on_step: bool = False,
        process_group: bool = None,
        dist_sync_fn: bool = None,
        use_cpu: bool = False,
        output_dir: str = None,
    ):
        """
        Args:
            subregions (bool, optional): If ``True``, femoral/tibial/patellar cartilage
                are divided into relevant subregions using DOSMA.
            channel_names (Sequence[str], optional): The ordered list of tissues to
                process. This should correspond to channels in the segmentation mask.
                Use ``'fc'`` for femoral cartilage, ``'tc'`` for tibial cartilage,
                ``'pc'`` for patellar cartilage, and ``'men'`` for meniscus.
            units (str, optional): Units for this metric.
        """
        if (
            subregions
            and channel_names
            and any(c not in self._SUPPORTED_TISSUES for c in channel_names)
        ):
            logger.warning(
                f"Channels {set(channel_names) - self._SUPPORTED_TISSUES} are not "
                f"supported for subdivision. Only averages will be reported"
            )
        self.categories = channel_names

        if isinstance(subregions, bool):
            subregions = channel_names if subregions else ()
        elif isinstance(subregions, str):
            subregions = (subregions,)
        unknown_subregions = set(subregions) - set(channel_names)
        if len(unknown_subregions) > 0:
            raise ValueError(f"Unknown subregions: {unknown_subregions}")
        self.subregions = subregions
        self.use_cpu = use_cpu
        self.output_dir = output_dir

        super().__init__(
            channel_names=None,
            units=units,
            reduction=reduction,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def update(
        self,
        quantitative_map: Sequence[dm.MedicalVolume] = None,
        sem_seg: Union[torch.Tensor, Sequence[dm.MedicalVolume]] = None,
        medial_direction: Union[str, Sequence[str]] = None,
        ids=None,
    ):
        assert quantitative_map is not None and sem_seg is not None
        if ids is None:
            ids = self._generate_ids(num_samples=len(quantitative_map))

        if isinstance(medial_direction, str):
            medial_direction = [medial_direction] * len(quantitative_map)
        for scan_id, qmap, seg, md in zip(ids, quantitative_map, sem_seg, medial_direction):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Mean of empty.*", category=RuntimeWarning
                )
                self.values.append(self._per_example_analysis(scan_id, qmap, seg, md))

        ids = self._add_ids(ids=ids, num_samples=len(quantitative_map))

    def _per_example_analysis(self, scan_id, qmap, sem_seg, medial_direction) -> Dict[str, float]:
        # TODO: Fix this field
        assert medial_direction in ("L", "R"), f"medial_direction={medial_direction}"
        use_largest_cc = True
        output_dir = self.output_dir

        if isinstance(sem_seg, torch.Tensor):
            sem_seg = dm.MedicalVolume.from_torch(
                sem_seg.permute(tuple(range(1, sem_seg.ndim)) + (0,)),  # channel last
                affine=qmap.affine,
            )
        assert isinstance(sem_seg, dm.MedicalVolume)

        # DOSMA has a bug where all scans get formatted to the SAGITTAL orientation,
        # where the last axis goes from Left -> Right (LR).
        # As a result, the scan goes from medial->lateral if it the medial direction
        # corresponds to the left side.
        sem_seg = sem_seg.reformat(SAGITTAL)
        assert sem_seg.orientation[-1] == "LR"
        medial_to_lateral = medial_direction == "L"

        qmap = qmap.reformat_as(sem_seg)
        if self.use_cpu:
            qmap = qmap.to("cpu")
            sem_seg = sem_seg.to("cpu")
        else:
            # cupyx.ndimage is faster on uint32 than bool
            sem_seg = sem_seg.astype(np.uint32)

        categories = {
            "fc": FemoralCartilage(medial_to_lateral=medial_to_lateral),
            "tc": TibialCartilage(medial_to_lateral=medial_to_lateral),
            "pc": PatellarCartilage(medial_to_lateral=medial_to_lateral),
            "men": Meniscus(medial_to_lateral=medial_to_lateral, split_ml_only=True),
        }
        all_data = []
        pbar = tqdm(self.categories, desc="", disable=True)
        for idx, tissue_key in enumerate(pbar):
            pbar.set_description(desc=f"Computing T2 {tissue_key}")
            df = pd.DataFrame()
            sem_seg_mv = sem_seg[..., idx]
            if self.subregions and tissue_key in self.subregions:
                tissue = categories.pop(tissue_key)
                kwargs = {
                    "use_largest_cc": use_largest_cc,
                    "use_largest_ccs": use_largest_cc,
                    "split_regions": tissue_key != "fc",
                }
                signature = inspect.signature(tissue.set_mask)
                kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
                tissue.set_mask(sem_seg_mv, **kwargs)

                # Compute regional quantitative analysis
                tissue.add_quantitative_value(T2(qmap))
                tissue.calc_quant_vals()
                if output_dir:
                    tissue.save_quant_data(os.path.join(output_dir, scan_id, tissue_key))
                df = tissue.quant_vals["T2"][1]
                df["Tissue"] = tissue_key
                region_keys = (
                    ["Location", "Condyle"]
                    if tissue_key == "pc"
                    else ["Location", "Side", "Region"]
                )
                if "Category" not in df.columns:
                    df["Category"] = (
                        tissue_key + "/" + df[region_keys].apply(lambda x: "-".join(x), axis=1)
                    )
                df = df[["Category", "Mean"]]

                # Clean up to avoid memory overhead
                del tissue

            # Compute full volume quantitative analysis
            df = df.append(
                {
                    "Category": tissue_key,
                    "Mean": np.nanmean(qmap.A[(qmap.A != 0) & (sem_seg_mv.A != 0)]).item(),
                },
                ignore_index=True,
            )
            all_data.append(df)
        del categories

        df = pd.concat(all_data, ignore_index=True)
        df = df.astype({"Mean": np.float64})
        df = pd.pivot_table(df, columns="Category", values="Mean", aggfunc=lambda x: x).reset_index(
            drop=True
        )
        return df

    def compute(self, reduction=None):
        if reduction is None:
            reduction = self.reduction
        return reduce(torch.as_tensor(pd.concat(self.values).to_numpy()), reduction)

    def _to_dict(self, device=None):
        values = (
            pd.concat(self.values, ignore_index=True)
            if isinstance(self.values, list)
            else self.values
        )
        channel_names = values.columns
        data = {"id": self.ids}
        data.update({name: torch.as_tensor(values[name].to_numpy()) for name in channel_names})
        return data
