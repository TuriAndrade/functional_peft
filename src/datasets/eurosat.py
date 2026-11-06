"""RGB EuroSAT with fixed stratified train/validation/test splits."""

import os
from pathlib import Path
from typing import Literal

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import EuroSAT

from .dataset import TorchDataset
from .transforms import PatchTransform


class EuroSATDataset(TorchDataset):
    """Stratified 80/10/10 partition of the full RGB EuroSAT dataset."""

    IMG_SIZE = 224
    N_CLASSES = 10
    N_TRAIN_SAMPLES = 21600
    N_VAL_SAMPLES = 2700
    N_TEST_SAMPLES = 2700
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
            root = Path(os.environ.get("DATA_DIR", "./")) / "eurosat"
        super().__init__(root=root, n_classes=self.N_CLASSES, download=download)

    def _build_splits(self):
        dataset = EuroSAT(root=str(self.root), download=self.download)
        if (
            len(dataset)
            != self.N_TRAIN_SAMPLES + self.N_VAL_SAMPLES + self.N_TEST_SAMPLES
        ):
            raise RuntimeError("Unexpected EuroSAT dataset size")

        labels = np.asarray(dataset.targets)
        train_idx, held_out_idx = self._stratified_split(
            np.arange(len(dataset)),
            labels,
            self.N_VAL_SAMPLES + self.N_TEST_SAMPLES,
            self.split_seed,
        )
        val_idx, test_idx = self._stratified_split(
            held_out_idx, labels, self.N_TEST_SAMPLES, self.split_seed
        )
        indices = {"train": train_idx, "val": val_idx, "test": test_idx}
        sources = {
            split: Subset(dataset, idx.tolist()) for split, idx in indices.items()
        }
        targets = {split: labels[idx] for split, idx in indices.items()}
        return sources, dataset.classes, targets

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
        transform = PatchTransform(
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
