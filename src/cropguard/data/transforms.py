"""Augmentation policies: light / medium / heavy.

MixUp/CutMix are batch-level and applied in the training loop (see training/module.py),
not here — this module only builds per-sample transforms.
"""

from __future__ import annotations

from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _base(image_size: int) -> list:
    import torch

    return [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def train_transforms(policy: str, image_size: int = 224) -> v2.Compose:
    if policy == "light":
        augs = [
            v2.RandomResizedCrop(image_size, scale=(0.8, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
        ]
    elif policy == "medium":
        augs = [
            v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            v2.RandomRotation(20),
        ]
    elif policy == "heavy":
        augs = [
            v2.RandomResizedCrop(image_size, scale=(0.6, 1.0), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandAugment(num_ops=2, magnitude=9),
        ]
    else:
        raise ValueError(f"Unknown augmentation policy: {policy!r}")

    transforms = v2.Compose(augs + _base(image_size))
    if policy == "heavy":
        transforms = v2.Compose(transforms.transforms + [v2.RandomErasing(p=0.25)])
    return transforms


def eval_transforms(image_size: int = 224) -> v2.Compose:
    resize = int(image_size * 256 / 224)
    return v2.Compose(
        [v2.Resize(resize, antialias=True), v2.CenterCrop(image_size)] + _base(image_size)
    )
