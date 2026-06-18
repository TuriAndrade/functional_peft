import os
from typing import Optional, Callable, List, Tuple, Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image


class FGVCNABirdsDataset(IterableDataset):
    """
    Streaming NABirds local-folder dataset.

    VPT-style FGVC setup:
      - Use the official NABirds train/test split from train_test_split.txt.
      - Since NABirds has no public validation split, split the official training
        set into:
            train : 90% of official train
            val   : 10% of official train
      - Use the official test split as test.
      - train+val uses the full official train split.

    Expected layout:
      nabirds/
        images/
          0295/
            *.jpg
          0296/
            *.jpg
          ...
        images.txt
        image_class_labels.txt
        train_test_split.txt
        classes.txt

    Expected sizes:
      - total: 48,562
      - official train: 23,929
      - train: 21,536
      - val: 2,393
      - train+val: 23,929
      - test: 24,633

    Notes:
      - This follows the VPT protocol, but not necessarily the exact original
        VPT sample identities, because the original VPT JSON split files are
        not used here.
      - The train/val split is deterministic using SPLIT_SEED.
      - Labels are remapped to contiguous [0, N_CLASSES - 1].
    """

    ROOT = os.path.join(
        os.environ.get("DATA_DIR", "./"),
        "fgvc",
        "nabirds",
    )
    IMAGE_DIR = "images"
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    IMG_SIZE = 224
    N_CLASSES = 555
    TOTAL_SAMPLES = 48562
    OFFICIAL_TRAIN_SAMPLES = 23929
    N_TRAIN_SAMPLES = 21536
    N_VAL_SAMPLES = 2393
    N_TRAIN_VAL_SAMPLES = 23929
    N_TEST_SAMPLES = 24633
    SPLIT_SEED = 0
    SHUFFLE_SEED = 0
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

        self.images_txt = os.path.join(self.ROOT, "images.txt")
        self.labels_txt = os.path.join(self.ROOT, "image_class_labels.txt")
        self.train_test_split_txt = os.path.join(self.ROOT, "train_test_split.txt")
        self.classes_txt = os.path.join(self.ROOT, "classes.txt")

        self._check_files()

        self.class_id_to_name = self._read_classes()

        official_train_samples, official_test_samples = self._read_official_samples()

        if len(official_train_samples) != self.OFFICIAL_TRAIN_SAMPLES:
            raise RuntimeError(
                f"Expected {self.OFFICIAL_TRAIN_SAMPLES} official NABirds train samples, "
                f"found {len(official_train_samples)}."
            )

        if len(official_test_samples) != self.N_TEST_SAMPLES:
            raise RuntimeError(
                f"Expected {self.N_TEST_SAMPLES} official NABirds test samples, "
                f"found {len(official_test_samples)}."
            )

        if (
            len(official_train_samples) + len(official_test_samples)
            != self.TOTAL_SAMPLES
        ):
            raise RuntimeError(
                f"Expected {self.TOTAL_SAMPLES} total NABirds samples, "
                f"found {len(official_train_samples) + len(official_test_samples)}."
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
            raise RuntimeError(f"Missing NABirds root directory: {self.ROOT}")

        if not os.path.isdir(self.image_root):
            raise RuntimeError(f"Missing NABirds image directory: {self.image_root}")

        required_files = [
            self.images_txt,
            self.labels_txt,
            self.train_test_split_txt,
            self.classes_txt,
        ]

        for path in required_files:
            if not os.path.isfile(path):
                raise RuntimeError(f"Missing required NABirds file: {path}")

    @staticmethod
    def _is_image_file(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in FGVCNABirdsDataset.IMAGE_EXTENSIONS

    def _read_classes(self) -> Dict[int, str]:
        """
        classes.txt contains the NABirds taxonomy entries.

        Important:
        - It can contain more entries than the actual classification labels.
        - The classification labels must be inferred from image_class_labels.txt.
        """

        id_to_class: Dict[int, str] = {}

        with open(self.classes_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                class_id_str, class_name = line.split(maxsplit=1)
                class_id = int(class_id_str)
                id_to_class[class_id] = class_name

        return id_to_class

    def _read_mapping_file(self, path: str) -> Dict[str, str]:
        """
        Generic parser for NABirds metadata files shaped like:
            <image_uuid> <value>

        Unlike CUB, NABirds image ids are UUID-like strings, not integers.
        """

        mapping: Dict[str, str] = {}

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                image_id, value = line.split(maxsplit=1)
                mapping[image_id] = value

        return mapping

    def _resolve_image_path(self, relpath: str) -> str:
        """
        NABirds images.txt can be handled robustly in either of these cases:

          1. relpath = "0295/xxx.jpg"
             -> ROOT/images/0295/xxx.jpg

          2. relpath = "images/0295/xxx.jpg"
             -> ROOT/images/0295/xxx.jpg
        """

        relpath = relpath.strip()

        candidate_from_image_root = os.path.join(self.image_root, relpath)
        if os.path.isfile(candidate_from_image_root):
            return candidate_from_image_root

        candidate_from_root = os.path.join(self.ROOT, relpath)
        if os.path.isfile(candidate_from_root):
            return candidate_from_root

        raise RuntimeError(
            f"Image listed in images.txt not found. Tried:\n"
            f"  {candidate_from_image_root}\n"
            f"  {candidate_from_root}"
        )

    def _read_official_samples(
        self,
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        Reads:
          - images.txt
          - image_class_labels.txt
          - train_test_split.txt

        Returns:
          - official_train_samples
          - official_test_samples

        Each sample is:
          (absolute_image_path, contiguous_label)
        """

        id_to_relpath = self._read_mapping_file(self.images_txt)
        id_to_raw_class_id_str = self._read_mapping_file(self.labels_txt)
        id_to_is_train_str = self._read_mapping_file(self.train_test_split_txt)

        image_ids = sorted(id_to_relpath)

        raw_class_ids = sorted(
            {int(id_to_raw_class_id_str[image_id]) for image_id in image_ids}
        )

        if len(raw_class_ids) != self.N_CLASSES:
            raise RuntimeError(
                f"Expected {self.N_CLASSES} unique NABirds class ids, "
                f"found {len(raw_class_ids)}."
            )

        raw_class_id_to_label = {
            raw_class_id: contiguous_label
            for contiguous_label, raw_class_id in enumerate(raw_class_ids)
        }

        official_train_samples: List[Tuple[str, int]] = []
        official_test_samples: List[Tuple[str, int]] = []

        for image_id in image_ids:
            relpath = id_to_relpath[image_id]
            raw_class_id = int(id_to_raw_class_id_str[image_id])
            label = raw_class_id_to_label[raw_class_id]

            is_train = int(id_to_is_train_str[image_id])
            image_path = self._resolve_image_path(relpath)

            if not self._is_image_file(image_path):
                raise RuntimeError(f"Non-image file listed as image: {image_path}")

            if not (0 <= label < self.N_CLASSES):
                raise RuntimeError(
                    f"Invalid label {label} for image_id={image_id}, relpath={relpath}"
                )

            sample = (image_path, label)

            if is_train == 1:
                official_train_samples.append(sample)
            elif is_train == 0:
                official_test_samples.append(sample)
            else:
                raise RuntimeError(
                    f"Invalid train_test_split value {is_train} for image_id={image_id}"
                )

        return official_train_samples, official_test_samples

    def _split_train_val(
        self,
        official_train_samples: List[Tuple[str, int]],
    ) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
        """
        VPT-style split:
        - use 90% of the official training set for train
        - use the remaining 10% for validation

        For NABirds:
        - official train = 23,929
        - train = int(0.90 * 23,929) = 21,536
        - val = 23,929 - 21,536 = 2,393

        The shuffle is applied over the official NABirds train list sorted by image id.
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
