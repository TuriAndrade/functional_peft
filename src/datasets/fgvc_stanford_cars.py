import os
from typing import Optional, Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat


class FGVCStanfordCarsDataset(IterableDataset):
    """
    Streaming Stanford Cars local-folder dataset.

    VPT-style FGVC setup:
      - Use the official Stanford Cars train/test split from:
            devkit/cars_train_annos.mat
            cars_test_annos_withlabels.mat
      - Since Stanford Cars has no public validation split, split the official
        training set into:
            train : 90% of official train
            val   : 10% of official train
      - Use the official test split as test.
      - train+val uses the full official train split.

    Expected layout:
      stanford_cars/
        cars_train/
          00001.jpg
          ...
        cars_test/
          00001.jpg
          ...
        cars_test_annos_withlabels.mat
        devkit/
          cars_train_annos.mat
          cars_meta.mat

    Expected sizes:
      - official train: 8,144
      - train: 7,329
      - val: 815
      - train+val: 8,144
      - test: 8,041

    Notes:
      - This follows the VPT protocol, but not necessarily the exact original
        VPT sample identities, because the original VPT JSON split files are
        unavailable.
      - The train/val split is deterministic using SPLIT_SEED.
    """

    ROOT = os.path.join(
        os.environ.get("DATA_DIR", "./"),
        "fgvc",
        "stanford_cars",
        "stanford_cars",
    )

    TRAIN_IMAGE_DIR = "cars_train"
    TEST_IMAGE_DIR = "cars_test"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    IMG_SIZE = 224
    N_CLASSES = 196
    OFFICIAL_TRAIN_SAMPLES = 8144
    N_TRAIN_SAMPLES = 7329
    N_VAL_SAMPLES = 815
    N_TEST_SAMPLES = 8041
    SPLIT_SEED = 0
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

        self.train_image_root = os.path.join(self.ROOT, self.TRAIN_IMAGE_DIR)
        self.test_image_root = os.path.join(self.ROOT, self.TEST_IMAGE_DIR)

        self.train_annos_mat = os.path.join(
            self.ROOT,
            "devkit",
            "cars_train_annos.mat",
        )
        self.test_annos_mat = os.path.join(
            self.ROOT,
            "cars_test_annos_withlabels.mat",
        )
        self.cars_meta_mat = os.path.join(
            self.ROOT,
            "devkit",
            "cars_meta.mat",
        )

        self._check_files()

        self.classes = self._read_classes()

        if len(self.classes) != self.N_CLASSES:
            raise RuntimeError(
                f"Expected {self.N_CLASSES} Stanford Cars classes, "
                f"found {len(self.classes)}."
            )

        official_train_samples = self._read_mat_samples(
            mat_path=self.train_annos_mat,
            image_root=self.train_image_root,
        )
        official_test_samples = self._read_mat_samples(
            mat_path=self.test_annos_mat,
            image_root=self.test_image_root,
        )

        if len(official_train_samples) != self.OFFICIAL_TRAIN_SAMPLES:
            raise RuntimeError(
                f"Expected {self.OFFICIAL_TRAIN_SAMPLES} official Stanford Cars "
                f"train samples, found {len(official_train_samples)}."
            )

        if len(official_test_samples) != self.N_TEST_SAMPLES:
            raise RuntimeError(
                f"Expected {self.N_TEST_SAMPLES} official Stanford Cars test samples, "
                f"found {len(official_test_samples)}."
            )

        train_samples, val_samples = self._split_train_val(official_train_samples)

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
            0 < n_test <= len(official_test_samples)
        ), f"n_test={n_test} is invalid. Must be in [1, {len(official_test_samples)}]"

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
            self.samples = official_test_samples[:n_test]

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
            raise RuntimeError(f"Missing Stanford Cars root directory: {self.ROOT}")

        if not os.path.isdir(self.train_image_root):
            raise RuntimeError(
                f"Missing Stanford Cars train image directory: {self.train_image_root}"
            )

        if not os.path.isdir(self.test_image_root):
            raise RuntimeError(
                f"Missing Stanford Cars test image directory: {self.test_image_root}"
            )

        required_files = [
            self.train_annos_mat,
            self.test_annos_mat,
            self.cars_meta_mat,
        ]

        for path in required_files:
            if not os.path.isfile(path):
                raise RuntimeError(f"Missing required Stanford Cars file: {path}")

    @staticmethod
    def _is_image_file(path: str) -> bool:
        return (
            os.path.splitext(path)[1].lower()
            in FGVCStanfordCarsDataset.IMAGE_EXTENSIONS
        )

    @staticmethod
    def _mat_to_str(value) -> str:
        """
        Converts MATLAB cell/string entries loaded by scipy.io.loadmat
        into a normal Python string.
        """

        while isinstance(value, np.ndarray):
            value = np.squeeze(value)

            if value.size != 1:
                value = value.flat[0]
            else:
                value = value.item()

        if isinstance(value, bytes):
            return value.decode("utf-8")

        return str(value)

    @staticmethod
    def _mat_to_int(value) -> int:
        """
        Converts MATLAB scalar entries loaded by scipy.io.loadmat
        into a normal Python int.
        """

        while isinstance(value, np.ndarray):
            value = np.squeeze(value)

            if value.size != 1:
                value = value.flat[0]
            else:
                value = value.item()

        return int(value)

    @staticmethod
    def _get_mat_field(annotation, field_name: str):
        """
        Robustly extracts a field from a MATLAB annotation struct.

        Works with scipy.io.loadmat(..., squeeze_me=True, struct_as_record=False).
        """

        if hasattr(annotation, field_name):
            return getattr(annotation, field_name)

        if isinstance(annotation, np.void) and annotation.dtype.names is not None:
            return annotation[field_name]

        if isinstance(annotation, np.ndarray) and annotation.dtype.names is not None:
            return annotation[field_name]

        raise RuntimeError(
            f"Could not find field '{field_name}' in Stanford Cars annotation."
        )

    def _read_classes(self) -> List[str]:
        """
        Reads Stanford Cars class names from devkit/cars_meta.mat.

        Expected key:
          - class_names
        """

        mat = loadmat(
            self.cars_meta_mat,
            squeeze_me=True,
            struct_as_record=False,
        )

        if "class_names" not in mat:
            raise RuntimeError(f"Missing key 'class_names' in {self.cars_meta_mat}")

        class_names_raw = np.atleast_1d(mat["class_names"]).ravel()
        class_names = [self._mat_to_str(class_name) for class_name in class_names_raw]

        return class_names

    def _read_mat_samples(
        self,
        mat_path: str,
        image_root: str,
    ) -> List[Tuple[str, int]]:
        """
        Reads Stanford Cars annotation .mat files.

        Expected key:
          - annotations

        Each annotation has fields:
          - fname: image filename
          - class: 1-based class label

        Returns:
          - samples: list of (absolute_image_path, contiguous_label)
        """

        mat = loadmat(
            mat_path,
            squeeze_me=True,
            struct_as_record=False,
        )

        if "annotations" not in mat:
            raise RuntimeError(f"Missing key 'annotations' in {mat_path}")

        annotations = np.atleast_1d(mat["annotations"]).ravel()

        samples: List[Tuple[str, int]] = []

        for annotation in annotations:
            fname_raw = self._get_mat_field(annotation, "fname")
            class_id_raw = self._get_mat_field(annotation, "class")

            fname = self._mat_to_str(fname_raw)
            class_id = self._mat_to_int(
                class_id_raw
            )  # Stanford Cars labels are 1-based.
            label = class_id - 1

            image_path = os.path.join(image_root, fname)

            if not os.path.isfile(image_path):
                raise RuntimeError(
                    f"Image listed in {os.path.basename(mat_path)} not found: "
                    f"{image_path}"
                )

            if not self._is_image_file(image_path):
                raise RuntimeError(f"Non-image file listed as image: {image_path}")

            if not (0 <= label < self.N_CLASSES):
                raise RuntimeError(
                    f"Invalid label {label} for fname={fname} in {mat_path}"
                )

            samples.append((image_path, label))

        return samples

    def _split_train_val(
        self,
        official_train_samples: List[Tuple[str, int]],
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        VPT-style split:
        - use 90% of the official training set for train
        - use the remaining 10% for validation

        For Stanford Cars:
        - official train = 8144
        - train = int(0.90 * 8144) = 7329
        - val = 8144 - 7329 = 815

        The shuffle is applied over the official Stanford Cars train annotation order.
        """

        n_total = len(official_train_samples)
        n_train = int(0.90 * n_total)
        n_val = n_total - n_train

        if n_train != self.N_TRAIN_SAMPLES or n_val != self.N_VAL_SAMPLES:
            raise RuntimeError(
                f"Unexpected regenerated split sizes: train={n_train}, val={n_val}. "
                f"Expected train={self.N_TRAIN_SAMPLES}, val={self.N_VAL_SAMPLES}."
            )

        rng = np.random.RandomState(self.SPLIT_SEED)
        indices = np.arange(n_total)
        rng.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_samples = [official_train_samples[i] for i in train_indices]
        val_samples = [official_train_samples[i] for i in val_indices]

        return train_samples, val_samples

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
