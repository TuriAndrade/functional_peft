"""Stanford Cars with fixed stratified train/validation splits."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import StanfordCars

from .dataset import TorchDataset
from .transforms import NaturalTransform


class StanfordCarsDataset(TorchDataset):
    """Official train/test with a stratified 90/10 training split."""

    IMG_SIZE = 224
    N_CLASSES = 196
    N_TRAIN_SAMPLES = 7329
    N_VAL_SAMPLES = 815
    N_TEST_SAMPLES = 8041
    SPLIT_SEED = 42

    def __init__(
        self,
        root: str | Path | None = None,
        img_size: int = IMG_SIZE,
        seed: int = SPLIT_SEED,
    ):
        self.img_size = img_size
        self.split_seed = seed
        if root is None:
            root = Path(os.environ.get("DATA_DIR", "./")) / "stanford_cars"
        super().__init__(root=root, n_classes=self.N_CLASSES, download=False)

    def _build_splits(self):
        train = StanfordCars(root=str(self.root), split="train", download=False)
        test = StanfordCars(root=str(self.root), split="test", download=False)
        if len(train) != self.N_TRAIN_SAMPLES + self.N_VAL_SAMPLES:
            raise RuntimeError("Unexpected Stanford Cars training size")
        if len(test) != self.N_TEST_SAMPLES:
            raise RuntimeError("Unexpected Stanford Cars test size")

        labels = np.asarray([label for _, label in train._samples])
        train_idx, val_idx = self._stratified_split(
            np.arange(len(train)), labels, self.N_VAL_SAMPLES, self.split_seed
        )
        sources = {
            "train": Subset(train, train_idx.tolist()),
            "val": Subset(train, val_idx.tolist()),
            "test": test,
        }
        targets = {
            "train": labels[train_idx],
            "val": labels[val_idx],
            "test": np.asarray([label for _, label in test._samples]),
        }
        return sources, train.classes, targets

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
