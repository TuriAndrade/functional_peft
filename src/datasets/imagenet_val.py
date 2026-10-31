import os
import random
from typing import Optional, Callable, List, Tuple, Dict

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image


class ImageNetValDataset(IterableDataset):
    """
    Streaming ImageNet-val local-folder dataset.

    Local ImageNet-val setup:
      - There is only one directory containing class subfolders.
      - The dataset is split deterministically into train/val/test.
      - Splitting is stratified per class.

    Default setup:
      - Use N_CLASSES = 100 classes sampled deterministically from the available
        ImageNet classes.
      - split='train'     : stratified sample from the train portion
      - split='val'       : stratified sample from the val portion
      - split='train+val' : train sample + val sample
      - split='test'      : stratified sample from the test portion

    Expected layout:
      imagenet_val_data_dir/
        class_name_or_wnid/
          image_1.JPEG
          image_2.JPEG
          ...

    Notes:
      - This is intended to characterize ImageNet-like behavior using ImageNet-val.
      - N_CLASSES can be changed, and class selection remains deterministic.
      - Train/val/test sampling is stratified.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG"}
    ROOT = os.path.join(
        os.environ.get("DATA_DIR", "./"),
        "imagenet-val",
    )

    IMG_SIZE = 224
    N_CLASSES = 100
    TRAIN_SPLIT_FRAC = 0.8
    VAL_SPLIT_FRAC = 0.1
    N_TRAIN_SAMPLES = 800
    N_VAL_SAMPLES = 200
    SPLIT_SEED = 0
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
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

        if not os.path.isdir(self.ROOT):
            raise RuntimeError(f"Missing ImageNet-val directory: {self.ROOT}")

        self.available_classes = self._get_available_classes()

        if len(self.available_classes) < self.N_CLASSES:
            raise RuntimeError(
                f"Expected at least {self.N_CLASSES} ImageNet classes, "
                f"found {len(self.available_classes)}."
            )

        self.classes = self._select_classes(self.available_classes)
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        train_by_class, val_by_class, test_by_class = self._collect_stratified_splits()

        train_count = sum(len(paths) for paths in train_by_class.values())
        val_count = sum(len(paths) for paths in val_by_class.values())
        test_count = sum(len(paths) for paths in test_by_class.values())

        n_train = self.N_TRAIN_SAMPLES if n_train is None else n_train
        n_val = self.N_VAL_SAMPLES if n_val is None else n_val
        n_test = test_count if n_test is None else n_test

        assert (
            0 < n_train <= train_count
        ), f"n_train={n_train} is invalid. Must be in [1, {train_count}]"
        assert (
            0 < n_val <= val_count
        ), f"n_val={n_val} is invalid. Must be in [1, {val_count}]"
        assert (
            0 < n_test <= test_count
        ), f"n_test={n_test} is invalid. Must be in [1, {test_count}]"

        self.split_counts = {
            "train": n_train,
            "val": n_val,
            "train+val": n_train + n_val,
            "test": n_test,
        }

        train_samples = self._take_stratified_samples(train_by_class, n_train)
        val_samples = self._take_stratified_samples(val_by_class, n_val)
        test_samples = self._take_stratified_samples(test_by_class, n_test)

        if self.split == "train":
            self.samples = train_samples

        elif self.split == "val":
            self.samples = val_samples

        elif self.split == "train+val":
            self.samples = train_samples + val_samples

        else:  # test
            self.samples = test_samples

        self.num_examples = len(self.samples)

    def get_split(self):
        return self.split

    def get_split_counts(self):
        return self.split_counts

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @staticmethod
    def _is_image_file(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in ImageNetValDataset.IMAGE_EXTENSIONS

    def _get_available_classes(self) -> List[str]:
        class_names = [entry.name for entry in os.scandir(self.ROOT) if entry.is_dir()]

        if not class_names:
            raise RuntimeError(f"No class directories found in {self.ROOT}")

        return sorted(class_names)

    def _select_classes(self, available_classes: List[str]) -> List[str]:
        """
        Select N_CLASSES deterministically from all available ImageNet classes.

        The subset is shuffled before selection to avoid always taking the
        alphabetically first classes. Labels are then assigned using the sorted
        selected class names for stability.
        """

        shuffled_classes = list(available_classes)

        rng = random.Random(self.SPLIT_SEED)
        rng.shuffle(shuffled_classes)

        selected_classes = shuffled_classes[: self.N_CLASSES]

        return sorted(selected_classes)

    def _collect_class_image_paths(self, class_name: str) -> List[str]:
        class_dir = os.path.join(self.ROOT, class_name)

        image_paths = []

        for dirpath, _, filenames in os.walk(class_dir):
            filenames = sorted(filenames)

            for filename in filenames:
                path = os.path.join(dirpath, filename)

                if self._is_image_file(path):
                    image_paths.append(path)

        image_paths.sort()

        if not image_paths:
            raise RuntimeError(f"No images found for ImageNet class: {class_name}")

        class_seed = self.SPLIT_SEED + sum(ord(ch) for ch in class_name)
        rng = random.Random(class_seed)
        rng.shuffle(image_paths)

        return image_paths

    def _collect_stratified_splits(
        self,
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Build train/val/test pools by splitting each class independently.

        This makes the base split stratified.
        """

        train_by_class: Dict[str, List[str]] = {}
        val_by_class: Dict[str, List[str]] = {}
        test_by_class: Dict[str, List[str]] = {}

        for class_name in self.classes:
            image_paths = self._collect_class_image_paths(class_name)

            n_total = len(image_paths)
            n_train = int(self.TRAIN_SPLIT_FRAC * n_total)
            n_val = int(self.VAL_SPLIT_FRAC * n_total)

            if n_train <= 0:
                raise RuntimeError(
                    f"Class {class_name} has too few images for train split: "
                    f"{n_total} images."
                )

            if n_val <= 0:
                raise RuntimeError(
                    f"Class {class_name} has too few images for val split: "
                    f"{n_total} images."
                )

            if n_total - n_train - n_val <= 0:
                raise RuntimeError(
                    f"Class {class_name} has too few images for test split: "
                    f"{n_total} images."
                )

            train_by_class[class_name] = image_paths[:n_train]
            val_by_class[class_name] = image_paths[n_train : n_train + n_val]
            test_by_class[class_name] = image_paths[n_train + n_val :]

        return train_by_class, val_by_class, test_by_class

    def _take_stratified_samples(
        self,
        samples_by_class: Dict[str, List[str]],
        n_samples: int,
    ) -> List[Tuple[str, int]]:
        """
        Take exactly n_samples from a split pool in a deterministic stratified way.

        The procedure is round-robin across classes:
          - class order is deterministically shuffled;
          - one image is taken from each class at a time;
          - this continues until n_samples are selected.

        This supports arbitrary N_CLASSES as long as the requested n_samples
        can be drawn from the corresponding split pool.
        """

        total_available = sum(len(paths) for paths in samples_by_class.values())

        assert (
            0 < n_samples <= total_available
        ), f"n_samples={n_samples} is invalid. Must be in [1, {total_available}]"

        class_order = list(self.classes)

        rng = random.Random(self.SPLIT_SEED + n_samples)
        rng.shuffle(class_order)

        selected: List[Tuple[str, int]] = []

        max_class_len = max(
            (len(paths) for paths in samples_by_class.values()),
            default=0,
        )

        for image_idx in range(max_class_len):
            for class_name in class_order:
                paths = samples_by_class[class_name]

                if image_idx >= len(paths):
                    continue

                label = self.class_to_idx[class_name]
                selected.append((paths[image_idx], label))

                if len(selected) == n_samples:
                    return selected

        return selected

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

        sample_indices = list(range(self.num_examples))

        if self.split in {"train", "train+val"}:
            rng = random.Random(self.SHUFFLE_SEED + self.epoch)
            rng.shuffle(sample_indices)

        for example_idx, sample_idx in enumerate(sample_indices):
            if example_idx % num_global_workers != global_worker_id:
                continue

            img_path, label = self.samples[sample_idx]

            img = Image.open(img_path).convert("RGB")
            img = self.TRANSFORM(img)

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
