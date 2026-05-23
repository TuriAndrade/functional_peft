import os
from typing import Optional, Callable

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image


class VTABCLEVRDistanceDataset(IterableDataset):
    """
    Streaming CLEVR/distance via TFDS.

    Official VTAB splits:
      - split='train'     : train[:800]
      - split='val'       : train[train_count:train_count+200]
      - split='train+val' : train[:800] + train[train_count:train_count+200]
      - split='test'      : validation

    VTAB label:
      - closest_distance = min(objects["pixel_coords"][:, 2])
      - bucket closest_distance using:

            [0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0]

      - this gives labels in [0, 5]
    """

    NAME = "clevr:3.*.*"
    DATA_DIR = os.path.join(os.environ.get("DATA_DIR", "./"), "vtab", "clevr_data_dir")
    EXTRACT_DIR = os.path.join(
        os.environ.get("DATA_DIR", "./"), "vtab", "clevr_extract_dir"
    )
    MANUAL_DIR = os.path.join(
        os.environ.get("DATA_DIR", "./"), "vtab", "clevr_manual_dir"
    )

    IMG_SIZE = 224
    N_CLASSES = 6
    TRAIN_SPLIT_FRAC = 0.9
    N_TRAIN_SAMPLES = 800
    N_VAL_SAMPLES = 200
    AS_SUPERVISED = False
    DISTANCE_THRESHOLDS = np.array(
        [0.0, 8.0, 8.5, 9.0, 9.5, 10.0, 100.0],
        dtype=np.float32,
    )
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
    SHUFFLE_BUFFER_SIZE = 1024

    def __init__(
        self,
        split: str,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        assert split in {"train", "val", "train+val", "test"}, f"Invalid split: {split}"

        self.split = split
        self.rank = rank
        self.world_size = world_size

        self.epoch = 0

        with tf.device("/CPU:0"):
            self.tfds_builder = tfds.builder(
                name=self.NAME,
                data_dir=self.DATA_DIR,
            )

            self.tfds_builder.download_and_prepare(
                download_config=tfds.download.DownloadConfig(
                    try_download_gcs=True,
                    manual_dir=self.MANUAL_DIR,
                    extract_dir=self.EXTRACT_DIR,
                )
            )

        trainval_count = self.tfds_builder.info.splits["train"].num_examples
        train_count = int(self.TRAIN_SPLIT_FRAC * trainval_count)

        # Official VTAB logical test split for CLEVR is TFDS "validation".
        test_count = self.tfds_builder.info.splits["validation"].num_examples

        if self.split == "train":
            self.tfds_split = f"train[:{self.N_TRAIN_SAMPLES}]"
            self.num_examples = self.N_TRAIN_SAMPLES

        elif self.split == "val":
            val_start = train_count
            val_end = train_count + self.N_VAL_SAMPLES

            self.tfds_split = f"train[{val_start}:{val_end}]"
            self.num_examples = self.N_VAL_SAMPLES

        elif self.split == "train+val":
            val_start = train_count
            val_end = train_count + self.N_VAL_SAMPLES

            self.tfds_split = (
                f"train[:{self.N_TRAIN_SAMPLES}]" f"+train[{val_start}:{val_end}]"
            )
            self.num_examples = self.N_TRAIN_SAMPLES + self.N_VAL_SAMPLES

        else:  # test
            self.tfds_split = "validation"
            self.num_examples = test_count

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        """
        Stream examples from TFDS.

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

        with tf.device("/CPU:0"):
            ds = self.tfds_builder.as_dataset(
                split=self.tfds_split,
                as_supervised=self.AS_SUPERVISED,
                shuffle_files=False,
            )

            if self.split in ["train", "train+val"]:
                ds = ds.shuffle(
                    buffer_size=min(self.num_examples, self.SHUFFLE_BUFFER_SIZE),
                    seed=self.SHUFFLE_SEED + self.epoch,
                    reshuffle_each_iteration=True,
                )

            # Small pipeline prefetch; this does not materialize the full dataset.
            ds = ds.prefetch(tf.data.AUTOTUNE)

        for example_idx, example in enumerate(tfds.as_numpy(ds)):
            # Prevent duplicate samples when using multiple workers / DDP.
            if example_idx % num_global_workers != global_worker_id:
                continue

            img_np = example["image"]
            pixel_coords = example["objects"]["pixel_coords"]

            if pixel_coords.ndim != 2 or pixel_coords.shape[1] < 3:
                raise ValueError(
                    f"Expected objects['pixel_coords'] with shape (num_objects, 3), "
                    f"got {pixel_coords.shape}."
                )

            # Official VTAB CLEVR/distance label:
            # closest object depth, bucketed into 6 distance classes.
            closest_distance = float(np.min(pixel_coords[:, 2]))

            label = int(
                np.max(np.where((self.DISTANCE_THRESHOLDS - closest_distance) < 0))
            )

            if not (0 <= label < self.N_CLASSES):
                raise ValueError(
                    f"Invalid CLEVR/distance label={label} from "
                    f"closest_distance={closest_distance}."
                )

            img = Image.fromarray(img_np)

            img = self.TRANSFORM(img)

            yield img, torch.tensor(label, dtype=torch.long)

    def get_dataloader(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        worker_init_fn=None,
        drop_last: bool = False,
        pin_memory: bool = True,
        collate_fn=None,
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
