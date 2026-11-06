"""GTSRB with physical-sign-track-disjoint train/validation splits."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import GTSRB

from .dataset import TorchDataset
from .transforms import SymbolTransform


class GTSRBDataset(TorchDataset):
    """Official train/test; validation holds out entire physical-sign tracks."""

    IMG_SIZE = 224
    N_CLASSES = 43
    N_TRAIN_VAL_SAMPLES = 26640
    VAL_SPLIT_FRAC = 0.1
    N_TEST_SAMPLES = 12630
    SPLIT_SEED = 42

    def __init__(
        self,
        root: str | Path | None = None,
        img_size: int = IMG_SIZE,
        seed: int = SPLIT_SEED,
        download: bool = True,
    ):
        self.img_size = img_size
        self.split_seed = seed
        if root is None:
            root = Path(os.environ.get("DATA_DIR", "./")) / "gtsrb"
        super().__init__(root=root, n_classes=self.N_CLASSES, download=download)

    def _build_splits(self):
        train = GTSRB(root=str(self.root), split="train", download=self.download)
        test = GTSRB(root=str(self.root), split="test", download=self.download)
        if len(train) != self.N_TRAIN_VAL_SAMPLES:
            raise RuntimeError("Unexpected GTSRB training size")
        if len(test) != self.N_TEST_SAMPLES:
            raise RuntimeError("Unexpected GTSRB test size")

        train_idx, val_idx = self._split_tracks(train._samples, self.split_seed)
        train_labels = np.asarray([label for _, label in train._samples])
        sources = {
            "train": Subset(train, train_idx.tolist()),
            "val": Subset(train, val_idx.tolist()),
            "test": test,
        }
        targets = {
            "train": train_labels[train_idx],
            "val": train_labels[val_idx],
            "test": np.asarray([label for _, label in test._samples]),
        }
        return sources, [str(i) for i in range(self.N_CLASSES)], targets

    @classmethod
    def _split_tracks(cls, samples, seed: int):
        """Hold out approximately 10% of each class's tracks for validation."""
        tracks_by_class = defaultdict(set)
        sample_keys = []
        for path, label in samples:
            # Each class contains filenames <track>_<frame>.ppm.
            parts = Path(path).stem.split("_")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"Unexpected GTSRB filename: {path}")
            key = (int(label), parts[0])
            tracks_by_class[int(label)].add(key)
            sample_keys.append(key)

        if set(tracks_by_class) != set(range(cls.N_CLASSES)):
            raise RuntimeError("Missing GTSRB classes in training data")

        rng = np.random.default_rng(seed)
        held_out = set()
        for label in range(cls.N_CLASSES):
            tracks = sorted(tracks_by_class[label])
            if len(tracks) < 2:
                raise RuntimeError(f"Too few GTSRB tracks in class {label}")
            rng.shuffle(tracks)
            n_val = min(
                len(tracks) - 1, max(1, round(cls.VAL_SPLIT_FRAC * len(tracks)))
            )
            held_out.update(tracks[:n_val])

        is_val = np.asarray([key in held_out for key in sample_keys], dtype=bool)
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)

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
        transform = SymbolTransform(
            mode=mode,
            norm=transform_norm,
            image_size=self.img_size,
            pad_to_square=True,
            color_jitter=True,
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
