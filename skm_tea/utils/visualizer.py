from enum import Enum
from typing import Dict, Sequence, Union

import meddlr.ops.complex as cplx
import numpy as np
import torch
import torchvision.utils as tv_utils


def _to_vis_tensor(image):
    """Convert to a visualization tensor."""
    if isinstance(image, torch.Tensor):
        return image.cpu()
    elif isinstance(image, np.ndarray) and image.dtype in (np.complex64, np.complex128):
        image = cplx.to_tensor(image)
        if cplx.is_complex(image):
            image = torch.view_as_real(image)
        return image
    elif isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    else:
        raise TypeError(f"`image` of type {type(image)} is not supported")


class VisImageMode(Enum):
    """Different image modes for tensors representing complex images.

    The complex dimension is 2.

    To add:
        keycode,
    """

    UNK = 0  # Unknown
    HW = 1  # Height x Width
    HW2 = 2  # Height x Width x 2
    NHW = 3  # Batch x Height x Width
    NHW2 = 4  # Batch x Height x Width x 2

    def __new__(cls, keycode):
        obj = object.__new__(cls)
        obj._value_ = keycode
        return obj

    def is_complex(self):
        return self in (VisImageMode.HW2, VisImageMode.NHW2)

    def ndim(self):
        return len(self.name)

    def vis_dim(self):
        _ndim = self.ndim
        if self.is_complex():
            _ndim -= 1
        return _ndim

    def abs(self, image):
        if self.is_complex() or cplx.is_complex(image) or cplx.is_complex_as_real(image):
            return cplx.abs(image)
        else:
            return torch.abs(image)


@torch.no_grad()
def draw_reconstructions(
    images: Sequence[Union[torch.Tensor, np.ndarray]],
    target=None,
    kspace=None,
    mode: Union[str, VisImageMode] = "auto",
    padding: int = 1,
    channels=None,
) -> Dict[str, np.ndarray]:
    """
    Note:
        Certain visualizations such as `phases` are only available with complex inputs.

    Args:
        images (torch.Tensor(s) or ndarrays(s)): Images to display such as reconstruction,
            zero-filled, etc. Shapes for all images should be (B,)H,W,(2,). If last dimension
        target (torch.Tensor or ndarray): The target image. This should follow the same
            shape and mode as `images`.
        kspace (torch.Tensor or ndarray): Last dimension must be 2.
        mode (Union[str, VisImageMode]): Defaults to `auto`, where last dimension will be
            interpreted as the complex dimension only if image is a tensor and last dimension
            has size 2.
        padding (int, optional): Padding to use between images.

    Returns:
        Dict[str, ndarray]: Arrays are in CHW order.
    """
    if isinstance(mode, str) and mode != "auto":
        raise ValueError("`mode` must be 'auto' or a VisImageMode")

    if not isinstance(images, (list, tuple)):
        images = [images]
    images = [_to_vis_tensor(image) for image in images]
    target = _to_vis_tensor(target) if target is not None else None
    kspace = _to_vis_tensor(kspace) if kspace is not None else None

    if channels:
        assert cplx.is_complex(images[0])
    nchannels = len(channels) if channels else 1

    if mode == "auto":
        if isinstance(images[0], torch.Tensor) and images[0].shape[-1] == 2:
            mode = VisImageMode.HW2 if images[0].ndim == 3 else VisImageMode.NHW2
        elif images[0].ndim == 2:
            mode = VisImageMode.HW
        elif images[0].ndim == 3:
            mode = VisImageMode.NHW
        else:
            raise ValueError("`mode` could not be automatically determined.")

    cat_dim = mode.name.index("W")  # cat along the width dimension
    all_images = images + [target] if target is not None else images
    all_images = torch.cat(all_images, dim=cat_dim)
    is_complex = mode.is_complex() or cplx.is_complex(all_images)

    imgs_to_write = {"images": mode.abs(all_images)}
    if target is not None:
        images_cat = torch.cat(images, dim=cat_dim)
        target_cat = torch.cat([target] * len(images), dim=cat_dim)
        imgs_to_write["errors"] = mode.abs(images_cat - target_cat)
    if is_complex:
        imgs_to_write["phases"] = cplx.angle(all_images)
    if kspace is not None:
        imgs_to_write["masks"] = cplx.get_mask(kspace).squeeze(-1)

    outputs = {}
    for name, data in imgs_to_write.items():
        if mode.name.startswith("N"):
            # Batch dimension already exists.
            if channels:
                data = torch.cat([data[..., i] for i in range(nchannels)], dim=0)
            data = data.unsqueeze(1)
        else:
            if channels:
                data = data.permute((data.ndim - 1,) + tuple(range(data.ndim - 1))).unsqueeze(1)
            else:
                data = data.unsqueeze(0).unsqueeze(0)

        data = tv_utils.make_grid(
            data, nrow=nchannels, padding=padding, normalize=True, scale_each=True
        )
        outputs[name] = data.numpy()
    return outputs
