"""Dataset splits, optional subsampling, and PyTorch data loading."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    Sampler,
    Subset,
)


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation examples across ranks without padding or duplication."""

    def __init__(self, dataset: Dataset, rank: int, world_size: int):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid rank or world_size")

        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))


class TransformedDataset(Dataset):
    """Transform images without modifying the underlying dataset."""

    def __init__(self, dataset: Dataset, transform: Callable | None = None):
        self.dataset = dataset
        self.transform = transform if transform is not None else torch.nn.Identity()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, *rest = self.dataset[index]
        return (self.transform(image), *rest)


class TorchDataset(ABC):
    """Manage complete dataset partitions and construct PyTorch DataLoaders."""

    def __init__(
        self,
        root: str | Path,
        n_classes: int,
        download: bool,
    ):
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")

        self.root = Path(root)
        self.n_classes = n_classes
        self.download = download

        sources, classes, targets = self._build_splits()
        required_splits = {"train", "val", "test"}

        if set(sources) != required_splits or set(targets) != required_splits:
            raise ValueError("_build_splits must provide train, val, and test")

        self.classes = list(classes)
        if len(self.classes) != n_classes:
            raise ValueError(f"Expected {n_classes} classes, got {len(self.classes)}")

        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self._splits = dict(sources)
        self._targets = {}

        for split in ("train", "val", "test"):
            labels = targets[split]
            if labels is not None:
                labels = np.asarray(labels)
                if len(labels) != len(self._splits[split]):
                    raise ValueError(f"Labels for {split} do not match dataset length")
            self._targets[split] = labels

        self.split_counts = {
            split: len(dataset) for split, dataset in self._splits.items()
        }
        self.split_counts["train+val"] = (
            self.split_counts["train"] + self.split_counts["val"]
        )

    @abstractmethod
    def _build_splits(
        self,
    ) -> tuple[
        Mapping[str, Dataset],
        Sequence[str],
        Mapping[str, Sequence[int] | None],
    ]:
        """Return complete, disjoint, untransformed splits and aligned labels.

        The returned mappings must contain train, val, and test. Labels may
        be None when stratified subsampling is unavailable.
        """
        raise NotImplementedError

    @staticmethod
    def _stratified_split(indices, labels, val_size: int, seed: int):
        """Split indices approximately preserving each class's proportions."""
        indices = np.asarray(indices)
        labels = np.asarray(labels)

        return train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            stratify=labels[indices],
        )

    @staticmethod
    def _sample_indices(
        size: int,
        n_samples: int | None,
        seed: int,
        labels=None,
    ) -> np.ndarray | None:
        """Select a reproducible subset; return None to use all examples."""
        if n_samples is None:
            return None

        if (
            isinstance(n_samples, (bool, np.bool_))
            or not isinstance(n_samples, (int, np.integer))
            or not 0 < n_samples <= size
        ):
            raise ValueError(f"n_samples must be in [1, {size}], got {n_samples}")

        if n_samples == size:
            return None

        indices = np.arange(size)
        if labels is not None:
            labels = np.asarray(labels)
            if len(labels) != size:
                raise ValueError("Labels do not match dataset length")

            _, counts = np.unique(labels, return_counts=True)
            n_classes = len(counts)
            if (
                np.all(counts >= 2)
                and n_samples >= n_classes
                and size - n_samples >= n_classes
            ):
                selected, _ = train_test_split(
                    indices,
                    train_size=int(n_samples),
                    random_state=seed,
                    stratify=labels,
                )
                return selected

        return np.random.default_rng(seed).permutation(size)[:n_samples]

    def get_dataset(
        self,
        split: str,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a transformed view of the complete requested partition."""
        if split not in {"train", "val", "train+val", "test"}:
            raise ValueError(f"Invalid split: {split}")

        source = (
            ConcatDataset([self._splits["train"], self._splits["val"]])
            if split == "train+val"
            else self._splits[split]
        )
        return TransformedDataset(source, transform)

    def get_split_counts(self) -> dict[str, int]:
        """Return the complete partition sizes, before any subsampling."""
        return self.split_counts.copy()

    def get_dataloader(
        self,
        split: str,
        transform: Callable | None = None,
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
        """Build a loader; call loader.sampler.set_epoch(epoch) for training."""
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid rank or world_size")

        dataset = self.get_dataset(split, transform=transform)

        if n_samples is not None:
            if split == "train+val":
                train_labels = self._targets["train"]
                val_labels = self._targets["val"]
                labels = (
                    np.concatenate([train_labels, val_labels])
                    if train_labels is not None and val_labels is not None
                    else None
                )
            else:
                labels = self._targets[split]

            indices = self._sample_indices(
                size=len(dataset),
                n_samples=n_samples,
                seed=seed,
                labels=labels,
            )
            if indices is not None:
                dataset = Subset(dataset, indices.tolist())

        training = split in {"train", "train+val"}
        shuffle = training if shuffle is None else shuffle

        if training:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=seed,
                drop_last=False,
            )
        elif world_size > 1:
            if shuffle:
                raise ValueError("Distributed evaluation cannot be shuffled")
            sampler = DistributedEvalSampler(
                dataset,
                rank=rank,
                world_size=world_size,
            )
        else:
            sampler = None

        generator = torch.Generator()
        generator.manual_seed(seed + rank)
        kwargs.setdefault("generator", generator)
        kwargs.setdefault("persistent_workers", num_workers > 0)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last if training else False,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            **kwargs,
        )
