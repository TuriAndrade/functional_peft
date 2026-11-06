"""Callable image transformations for training and evaluation."""

from typing import Literal

import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


NORMALIZATIONS = {
    "inception": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "clip": (
        (0.48145466, 0.45782750, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
}


def _validate_config(mode: str, norm: str, image_size: int):
    if mode not in {"stochastic", "deterministic"}:
        raise ValueError(f"Unknown transform mode: {mode!r}")
    if norm not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {norm!r}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")


class ToRGB:
    """Convert a PIL image to RGB."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")


class PadToSquare:
    """Center-pad a PIL image to a square without changing its aspect ratio."""

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        left = (side - width) // 2
        top = (side - height) // 2
        right = side - width - left
        bottom = side - height - top
        return ImageOps.expand(image, (left, top, right, bottom), fill=(0, 0, 0))


class RandomRotate90:
    """Rotate by a random multiple of 90 degrees without resampling."""

    def __call__(self, image: Image.Image) -> Image.Image:
        k = int(torch.randint(0, 4, (1,)).item())
        if k == 0:
            return image
        rotation = (
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )[k - 1]
        return image.transpose(rotation)


class NaturalTransform:
    """Crops and horizontal flips for natural-image classification."""

    def __init__(
        self,
        mode: Literal["stochastic", "deterministic"],
        norm: str = "inception",
        image_size: int = 224,
    ):
        _validate_config(mode, norm, image_size)
        interpolation = InterpolationMode.BICUBIC
        steps = [ToRGB()]

        if mode == "stochastic":
            steps.extend(
                [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 4 / 3),
                        interpolation=interpolation,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            steps.extend(
                [
                    transforms.Resize(
                        round(image_size * 256 / 224),
                        interpolation=interpolation,
                    ),
                    transforms.CenterCrop(image_size),
                ]
            )

        mean, std = NORMALIZATIONS[norm]
        steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.transform = transforms.Compose(steps)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class PatchTransform:
    """Right-angle rotations and flips with full-image resizing."""

    def __init__(
        self,
        mode: Literal["stochastic", "deterministic"],
        norm: str = "inception",
        image_size: int = 224,
    ):
        _validate_config(mode, norm, image_size)
        interpolation = InterpolationMode.BICUBIC
        steps = [ToRGB()]

        if mode == "stochastic":
            steps.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    RandomRotate90(),
                ]
            )

        steps.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=interpolation,
            )
        )
        mean, std = NORMALIZATIONS[norm]
        steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.transform = transforms.Compose(steps)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)


class SymbolTransform:
    """Mild affine augmentation and full-image resizing for signs and digits."""

    def __init__(
        self,
        mode: Literal["stochastic", "deterministic"],
        norm: str = "inception",
        image_size: int = 224,
        pad_to_square: bool = False,
        color_jitter: bool = False,
    ):
        _validate_config(mode, norm, image_size)
        interpolation = InterpolationMode.BICUBIC
        steps = [ToRGB()]

        if pad_to_square:
            steps.append(PadToSquare())

        if mode == "stochastic":
            steps.append(
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    interpolation=interpolation,
                    fill=(0, 0, 0),
                )
            )

        steps.append(
            transforms.Resize(
                (image_size, image_size),
                interpolation=interpolation,
            )
        )
        if mode == "stochastic" and color_jitter:
            steps.append(transforms.ColorJitter(brightness=0.1, contrast=0.1))

        mean, std = NORMALIZATIONS[norm]
        steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.transform = transforms.Compose(steps)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image)
