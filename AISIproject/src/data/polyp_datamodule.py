from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangePercentilesd,
)

LOGGER = logging.getLogger(__name__)


def _pair_images_masks(split_dir: Path, image_suffix: str, mask_suffix: str) -> List[Dict[str, Path]]:
    """Pair images with masks assuming split_dir/images and split_dir/masks share stem names."""
    image_root = split_dir / "images"
    mask_root = split_dir / "masks"
    if not image_root.exists() or not mask_root.exists():
        raise FileNotFoundError(
            f"Expected split_dir to contain 'images' and 'masks' subfolders: {split_dir}"
        )

    pairs: List[Dict[str, Path]] = []
    for img_path in sorted(image_root.rglob(f"*{image_suffix}")):
        mask_path = mask_root / f"{img_path.stem}{mask_suffix}"
        if not mask_path.exists():
            LOGGER.warning("Mask not found for %s, skipping", img_path.name)
            continue
        pairs.append({"image": img_path, "mask": mask_path})

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {split_dir}")
    return pairs


def _build_transforms(img_size: Tuple[int, int], is_train: bool):
    """Compose transforms for training or validation."""
    base = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),  # make channels-first for MONAI
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0.5, upper=99.5, b_min=0.0, b_max=1.0, clip=True
        ),
        ResizeWithPadOrCropd(keys=["image", "mask"], spatial_size=img_size),
        EnsureTyped(keys=["image", "mask"]),
    ]

    if is_train:
        aug = [
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1),
            RandRotated(
                keys=["image", "mask"],
                prob=0.3,
                range_x=0.1,
                range_y=0.1,
                padding_mode="border",
            ),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
        ]
        base = base + aug

    base.append(AsDiscrete(keys=["mask"], to_onehot=None))
    return Compose(base)


def create_datasets(
    train_dir: str,
    val_dir: str,
    image_suffix: str,
    mask_suffix: str,
    image_size: Tuple[int, int],
    cache_rate: float = 0.1,
) -> Tuple[CacheDataset, CacheDataset]:
    """Create MONAI CacheDatasets for train/val splits."""
    train_pairs = _pair_images_masks(Path(train_dir), image_suffix, mask_suffix)
    val_pairs = _pair_images_masks(Path(val_dir), image_suffix, mask_suffix)

    train_ds = CacheDataset(
        data=train_pairs,
        transform=_build_transforms(image_size, is_train=True),
        cache_rate=cache_rate,
    )
    val_ds = CacheDataset(
        data=val_pairs,
        transform=_build_transforms(image_size, is_train=False),
        cache_rate=min(cache_rate, 0.05),
    )
    return train_ds, val_ds


def create_dataloaders(
    train_dir: str,
    val_dir: str,
    image_suffix: str = ".jpg",
    mask_suffix: str = ".png",
    image_size: Tuple[int, int] = (512, 512),
    cache_rate: float = 0.1,
    batch_size: int = 2,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/val dataloaders ready for training."""
    train_ds, val_ds = create_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        image_suffix=image_suffix,
        mask_suffix=mask_suffix,
        image_size=image_size,
        cache_rate=cache_rate,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
