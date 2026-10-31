import os
from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat


class FGVCOxfordFlowersDataset(IterableDataset):
    """
    Streaming Oxford Flowers102 local-folder dataset.

    VPT-style FGVC setup:
      - Use the official Oxford Flowers102 splits from setid.mat.
      - Since Flowers102 has a public validation split, DO NOT create a
        regenerated 90/10 train/val split.
      - train+val uses the official train + official val splits.

    Expected layout:
      oxford_flowers102/
        jpg/
          image_00001.jpg
          image_00002.jpg
          ...
        imagelabels.mat
        setid.mat

    Expected sizes:
      - train: 1,020
      - val: 1,020
      - train+val: 2,040
      - test: 6,149

    Notes:
      - image ids in setid.mat are 1-based.
      - labels in imagelabels.mat are 1-based and are converted to 0-based.
    """

    ROOT = os.path.join(
        os.environ.get("DATA_DIR", "./"),
        "fgvc",
        "oxford_flowers102",
    )
    IMAGE_DIR = "jpg"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    IMG_SIZE = 224
    N_CLASSES = 102
    N_TRAIN_SAMPLES = 1020
    N_VAL_SAMPLES = 1020
    N_TRAIN_VAL_SAMPLES = 2040
    N_TEST_SAMPLES = 6149
    TRAIN_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    EVAL_TRANSFORM = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )
    SHUFFLE_SEED = 0

    def __init__(
        self,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        n_test: Optional[int] = None,
    ):
        super().__init__()

        assert split in {"train", "val", "train+val", "test"}, f"Invalid split: {split}"

        self.split = split
        self.rank = rank
        self.world_size = world_size

        self.epoch = 0

        self.image_root = os.path.join(self.ROOT, self.IMAGE_DIR)

        self.labels_mat = os.path.join(self.ROOT, "imagelabels.mat")
        self.setid_mat = os.path.join(self.ROOT, "setid.mat")

        self._check_files()

        train_samples, val_samples, test_samples = self._read_official_samples()

        if len(train_samples) != self.N_TRAIN_SAMPLES:
            raise RuntimeError(
                f"Expected {self.N_TRAIN_SAMPLES} Oxford Flowers train samples, "
                f"found {len(train_samples)}."
            )

        if len(val_samples) != self.N_VAL_SAMPLES:
            raise RuntimeError(
                f"Expected {self.N_VAL_SAMPLES} Oxford Flowers val samples, "
                f"found {len(val_samples)}."
            )

        if len(test_samples) != self.N_TEST_SAMPLES:
            raise RuntimeError(
                f"Expected {self.N_TEST_SAMPLES} Oxford Flowers test samples, "
                f"found {len(test_samples)}."
            )

        n_train = self.N_TRAIN_SAMPLES if n_train is None else n_train
        n_val = self.N_VAL_SAMPLES if n_val is None else n_val
        n_test = self.N_TEST_SAMPLES if n_test is None else n_test

        assert (
            0 < n_train <= len(train_samples)
        ), f"n_train={n_train} is invalid. Must be in [1, {len(train_samples)}]"
        assert (
            0 < n_val <= len(val_samples)
        ), f"n_val={n_val} is invalid. Must be in [1, {len(val_samples)}]"
        assert (
            0 < n_test <= len(test_samples)
        ), f"n_test={n_test} is invalid. Must be in [1, {len(test_samples)}]"

        self.split_counts = {
            "train": n_train,
            "val": n_val,
            "train+val": n_train + n_val,
            "test": n_test,
        }

        if self.split == "train":
            self.samples = train_samples[:n_train]

        elif self.split == "val":
            self.samples = val_samples[:n_val]

        elif self.split == "train+val":
            self.samples = train_samples[:n_train] + val_samples[:n_val]

        else:  # test
            self.samples = test_samples[:n_test]

        self.num_examples = len(self.samples)

        self.transform = (
            self.TRAIN_TRANSFORM
            if self.split in {"train", "train+val"}
            else self.EVAL_TRANSFORM
        )

    def get_split(self):
        return self.split

    def get_split_counts(self):
        return self.split_counts

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _check_files(self):
        if not os.path.isdir(self.ROOT):
            raise RuntimeError(f"Missing Oxford Flowers root directory: {self.ROOT}")

        if not os.path.isdir(self.image_root):
            raise RuntimeError(
                f"Missing Oxford Flowers image directory: {self.image_root}"
            )

        required_files = [
            self.labels_mat,
            self.setid_mat,
        ]

        for path in required_files:
            if not os.path.isfile(path):
                raise RuntimeError(f"Missing required Oxford Flowers file: {path}")

    @staticmethod
    def _is_image_file(path: str) -> bool:
        return (
            os.path.splitext(path)[1].lower()
            in FGVCOxfordFlowersDataset.IMAGE_EXTENSIONS
        )

    def _image_id_to_path(self, image_id: int) -> str:
        """
        Oxford Flowers image names are:
            image_00001.jpg
            image_00002.jpg
            ...
        """

        filename = f"image_{image_id:05d}.jpg"
        return os.path.join(self.image_root, filename)

    def _build_samples_from_ids(
        self,
        image_ids: np.ndarray,
        labels: np.ndarray,
    ) -> List[Tuple[str, int]]:
        """
        Builds samples from 1-based image ids.

        Each sample is:
          (absolute_image_path, contiguous_label)
        """

        samples: List[Tuple[str, int]] = []

        for image_id in image_ids:
            image_id = int(image_id)

            # labels array is indexed 0-based, but image ids are 1-based.
            label = int(labels[image_id - 1]) - 1

            image_path = self._image_id_to_path(image_id)

            if not os.path.isfile(image_path):
                raise RuntimeError(f"Image listed in setid.mat not found: {image_path}")

            if not self._is_image_file(image_path):
                raise RuntimeError(f"Non-image file listed as image: {image_path}")

            if not (0 <= label < self.N_CLASSES):
                raise RuntimeError(f"Invalid label {label} for image_id={image_id}")

            samples.append((image_path, label))

        return samples

    def _read_official_samples(
        self,
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Reads:
          - imagelabels.mat
          - setid.mat

        Returns:
          - official_train_samples
          - official_val_samples
          - official_test_samples

        Each sample is:
          (absolute_image_path, contiguous_label)
        """

        labels_data = loadmat(self.labels_mat)
        setid_data = loadmat(self.setid_mat)

        labels = labels_data["labels"].reshape(-1)

        train_ids = setid_data["trnid"].reshape(-1)
        val_ids = setid_data["valid"].reshape(-1)
        test_ids = setid_data["tstid"].reshape(-1)

        train_samples = self._build_samples_from_ids(train_ids, labels)
        val_samples = self._build_samples_from_ids(val_ids, labels)
        test_samples = self._build_samples_from_ids(test_ids, labels)

        return train_samples, val_samples, test_samples

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        """
        Stream local image examples.

        Handles:
          - DataLoader num_workers > 1
          - DDP rank/world_size
        """

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        global_worker_id = self.rank * num_workers + worker_id
        num_global_workers = self.world_size * num_workers

        sample_indices = np.arange(self.num_examples)

        if self.split in {"train", "train+val"}:
            rng = np.random.RandomState(self.SHUFFLE_SEED + self.epoch)
            rng.shuffle(sample_indices)

        for example_idx, sample_idx in enumerate(sample_indices):
            if example_idx % num_global_workers != global_worker_id:
                continue

            img_path, label = self.samples[sample_idx]

            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)

            yield img, torch.tensor(label, dtype=torch.long)

    def get_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        worker_init_fn: Optional[Callable] = None,
        drop_last: bool = False,
        pin_memory: bool = True,
        collate_fn: Optional[Callable] = None,
        *dl_args,
        **dl_kwargs,
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            *dl_args,
            **dl_kwargs,
        )
