import os
import random
from typing import Optional, Callable, List, Tuple, Dict

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image


class VTABSun397Dataset(IterableDataset):
    """
    Streaming SUN397 local-folder dataset.

    Approximation of VTAB SUN397 using only local train/ and test/ folders.

    Official VTAB SUN397 splits:
      - train      : train[:800]
      - val        : validation[:200]
      - train+val  : train[:800] + validation[:200]
      - test       : test

    Local approximation used here:
      - split='train'     : first 800 samples from shuffled/interleaved local train/
      - split='val'       : next 200 samples from shuffled/interleaved local train/
      - split='train+val' : first 1000 samples from shuffled/interleaved local train/
      - split='test'      : all samples from shuffled/interleaved local test/

    Expected layout:
      sun397_data_dir/
        train/
          class_name/
            ... optional nested subdirs ...
              image.jpg
        test/
          class_name/
            ... optional nested subdirs ...
              image.jpg

    Notes:
      - This does not exactly reproduce official VTAB SUN397 because local
        Kaggle version has no validation/ split.
      - To avoid taking only alphabetically early classes, local samples are
        ordered by deterministic class interleaving:
            first image of each shuffled class, second image of each shuffled class, ...
      - Class order and image order are shuffled deterministically with SPLIT_SEED.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ROOT = os.path.join(os.environ.get("DATA_DIR", "./"), "vtab", "sun397_data_dir")
    TRAIN_DIR = "train"
    TEST_DIR = "test"

    IMG_SIZE = 224
    N_CLASSES = 397
    TRAIN_SPLIT_FRAC = 0.9
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

        self.train_root = os.path.join(self.ROOT, self.TRAIN_DIR)
        self.test_root = os.path.join(self.ROOT, self.TEST_DIR)

        if not os.path.isdir(self.train_root):
            raise RuntimeError(f"Missing train directory: {self.train_root}")

        if not os.path.isdir(self.test_root):
            raise RuntimeError(f"Missing test directory: {self.test_root}")

        # Stable canonical class order for labels.
        self.classes = self._get_classes()
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}

        # Separate shuffled class order only for interleaving.
        self.interleaving_classes = self._get_shuffled_classes(self.classes)

        if len(self.classes) != self.N_CLASSES:
            raise RuntimeError(
                f"Expected {self.N_CLASSES} SUN397 classes, found {len(self.classes)}. "
                "Check whether your folder layout is root/split/class_name/images."
            )

        train_samples = self._collect_split_samples(self.train_root)
        test_samples = self._collect_split_samples(self.test_root)
        train_count = int(self.TRAIN_SPLIT_FRAC * len(train_samples))

        n_train = self.N_TRAIN_SAMPLES if n_train is None else n_train
        n_val = self.N_VAL_SAMPLES if n_val is None else n_val
        n_test = len(test_samples) if n_test is None else n_test

        assert (
            0 < n_train <= train_count
        ), f"n_train={n_train} is invalid. Must be in [1, {train_count}]"
        assert (
            0 < n_val <= len(train_samples) - train_count
        ), f"n_val={n_val} is invalid. Must be in [1, {len(train_samples) - train_count}]"
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
            val_start = train_count
            val_end = train_count + n_val

            self.samples = train_samples[val_start:val_end]

        elif self.split == "train+val":
            val_start = train_count
            val_end = train_count + n_val
            self.samples = train_samples[:n_train] + train_samples[val_start:val_end]

        else:  # test
            self.samples = test_samples[:n_test]

        self.num_examples = len(self.samples)

    def get_split(self):
        return self.split

    def get_split_counts(self):
        return self.split_counts

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    @staticmethod
    def _is_image_file(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in VTABSun397Dataset.IMAGE_EXTENSIONS

    def _get_classes(self) -> List[str]:
        class_names = set()

        for split_root in [self.train_root, self.test_root]:
            for entry in os.scandir(split_root):
                if entry.is_dir():
                    class_names.add(entry.name)

        if not class_names:
            raise RuntimeError(f"No class directories found in {self.ROOT}")

        return sorted(class_names)

    def _get_shuffled_classes(self, classes: List[str]) -> List[str]:
        shuffled_classes = list(classes)

        rng = random.Random(self.SPLIT_SEED)
        rng.shuffle(shuffled_classes)

        return shuffled_classes

    def _collect_class_image_paths(self, class_dir: str, class_name: str) -> List[str]:
        image_paths = []

        for dirpath, _, filenames in os.walk(class_dir):
            filenames = sorted(filenames)

            for filename in filenames:
                path = os.path.join(dirpath, filename)

                if self._is_image_file(path):
                    image_paths.append(path)

        image_paths.sort()

        # Deterministic per-class image shuffle.
        # The class_name-dependent offset avoids every class receiving the exact
        # same permutation pattern when class sizes are similar.
        class_seed = self.SPLIT_SEED + sum(ord(ch) for ch in class_name)
        rng = random.Random(class_seed)
        rng.shuffle(image_paths)

        return image_paths

    def _collect_split_samples(self, split_root: str) -> List[Tuple[str, int]]:
        """
        Collect samples in deterministic shuffled class-interleaved order.

        This means:
          - shuffle classes deterministically;
          - shuffle images inside each class deterministically;
          - take image 0 from shuffled class 0, image 0 from shuffled class 1, ...
          - then image 1 from shuffled class 0, image 1 from shuffled class 1, ...
          - and so on.

        Labels are still based on sorted class order, not shuffled class order.
        """

        per_class_paths: Dict[str, List[str]] = {}

        for class_name in self.interleaving_classes:
            class_dir = os.path.join(split_root, class_name)

            if not os.path.isdir(class_dir):
                per_class_paths[class_name] = []
                continue

            per_class_paths[class_name] = self._collect_class_image_paths(
                class_dir=class_dir,
                class_name=class_name,
            )

        max_class_len = max(
            (len(paths) for paths in per_class_paths.values()),
            default=0,
        )

        samples: List[Tuple[str, int]] = []

        for image_idx in range(max_class_len):
            for class_name in self.interleaving_classes:
                paths = per_class_paths[class_name]

                if image_idx >= len(paths):
                    continue

                label = self.class_to_idx[class_name]
                samples.append((paths[image_idx], label))

        return samples

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
