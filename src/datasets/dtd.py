"""Describable Textures Dataset with official partitions."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import DTD

from .dataset import TorchDataset
from .transforms import NaturalTransform


class DTDDataset(TorchDataset):
    """The three official DTD splits for one of its ten partitions."""

    IMG_SIZE = 224
    N_CLASSES = 47
    N_TRAIN_SAMPLES = 1880
    N_VAL_SAMPLES = 1880
    N_TEST_SAMPLES = 1880
    SPLIT_SEED = 42

    def __init__(
        self,
        root: str | Path | None = None,
        img_size: int = IMG_SIZE,
        seed: int = SPLIT_SEED,
        partition: int = 1,
        download: bool = True,
    ):
        if type(partition) is not int or not 1 <= partition <= 10:
            raise ValueError("partition must be an integer from 1 to 10")

        self.img_size = img_size
        self.split_seed = seed  # Official DTD splits do not depend on the seed.
        self.partition = partition
        if root is None:
            root = Path(os.environ.get("DATA_DIR", "./")) / "dtd"
        super().__init__(root=root, n_classes=self.N_CLASSES, download=download)

    def _build_splits(self):
        sources = {
            split: DTD(
                root=str(self.root),
                split=split,
                partition=self.partition,
                download=self.download if split == "train" else False,
            )
            for split in ("train", "val", "test")
        }
        for split, expected in (
            ("train", self.N_TRAIN_SAMPLES),
            ("val", self.N_VAL_SAMPLES),
            ("test", self.N_TEST_SAMPLES),
        ):
            if len(sources[split]) != expected:
                raise RuntimeError(f"Unexpected DTD {split} size")
            if sources[split].classes != sources["train"].classes:
                raise RuntimeError(f"Inconsistent DTD classes in {split}")

        targets = {
            split: np.asarray(dataset._labels) for split, dataset in sources.items()
        }
        return sources, sources["train"].classes, targets

    def get_dataloader(
        self,
        split: str,
        transform_mode: Literal["stochastic", "deterministic"] | None = None,
        transform_norm: str = "inception",
        n_samples: int | None = None,
        batch_size: int = 64,
        num_workers: int = 4,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool | None = None,
        worker_init_fn=None,
        drop_last: bool = False,
        pin_memory: bool = True,
        collate_fn=None,
        **kwargs,
    ) -> DataLoader:
        """Choose image preprocessing and delegate loading to TorchDataset."""
        mode = (
            transform_mode
            if transform_mode is not None
            else ("stochastic" if split in {"train", "train+val"} else "deterministic")
        )
        transform = NaturalTransform(
            mode=mode,
            norm=transform_norm,
            image_size=self.img_size,
        )
        return super().get_dataloader(
            split=split,
            transform=transform,
            n_samples=n_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            seed=seed,
            shuffle=shuffle,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **kwargs,
        )
