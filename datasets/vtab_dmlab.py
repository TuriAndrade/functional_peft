import os
from io import BytesIO

import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download


class VTABDMLabDataset(IterableDataset):
    """
    DMLab from Hugging Face mini-VTAB, avoiding `from datasets import load_dataset`
    because this repo already has a local package named `datasets`.

    Source:
      - repo: antofuller/mini-VTAB
      - files:
          dmlab/train-00000-of-00001.parquet
          dmlab/test-00000-of-00001.parquet
      - DMLab is not available at TFDS anymore.

    Exposed VTAB-like splits:
      - split='train'     : HF train[:800]
      - split='val'       : HF train[800:1000]
      - split='train+val' : HF train[:1000]
      - split='test'      : HF test[:]

    Labels:
      - Labels are already provided in the parquet file.
      - No custom label computation is performed.
      - DMLab has 6 classes.
    """

    HF_REPO_ID = "antofuller/mini-VTAB"
    HF_SUBDIR = "dmlab"
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    DATA_DIR = os.path.join(os.environ.get("DATA_DIR", "./"), "vtab", "dmlab_data_dir")

    IMG_SIZE = 224
    N_CLASSES = 6
    N_TRAIN_SAMPLES = 800
    N_VAL_SAMPLES = 200
    N_TRAINVAL_SAMPLES = 1000
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
    ):
        super().__init__()

        assert split in {"train", "val", "train+val", "test"}, f"Invalid split: {split}"

        self.split = split
        self.rank = rank
        self.world_size = world_size

        self.epoch = 0

        os.makedirs(self.DATA_DIR, exist_ok=True)

        if self.split == "train":
            self.hf_base_split = "train"
            self.start_idx = 0
            self.end_idx = self.N_TRAIN_SAMPLES

        elif self.split == "val":
            self.hf_base_split = "train"
            self.start_idx = self.N_TRAIN_SAMPLES
            self.end_idx = self.N_TRAIN_SAMPLES + self.N_VAL_SAMPLES

        elif self.split == "train+val":
            self.hf_base_split = "train"
            self.start_idx = 0
            self.end_idx = self.N_TRAINVAL_SAMPLES

        else:  # test
            self.hf_base_split = "test"
            self.start_idx = None
            self.end_idx = None

        parquet_filename = (
            f"{self.HF_SUBDIR}/{self.hf_base_split}-00000-of-00001.parquet"
        )

        self.parquet_path = hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=parquet_filename,
            repo_type="dataset",
            cache_dir=self.DATA_DIR,
            token=self.HF_TOKEN,
        )

        self.dataframe = pd.read_parquet(self.parquet_path)

        if self.start_idx is not None or self.end_idx is not None:
            self.dataframe = self.dataframe.iloc[self.start_idx : self.end_idx]

        self.dataframe = self.dataframe.reset_index(drop=True)
        self.num_examples = len(self.dataframe)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.num_examples

    @staticmethod
    def _decode_image(image_value):
        """
        Decode the HF parquet Image column.

        Depending on pyarrow/pandas versions, the image field may appear as:
          - a dict like {'bytes': ..., 'path': ...}
          - raw bytes
          - a PIL image
        """

        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")

        if isinstance(image_value, dict):
            image_bytes = image_value.get("bytes", None)
            image_path = image_value.get("path", None)

            if image_bytes is not None:
                return Image.open(BytesIO(image_bytes)).convert("RGB")

            if image_path is not None:
                return Image.open(image_path).convert("RGB")

            raise ValueError(
                f"Could not decode image dict. Keys: {list(image_value.keys())}"
            )

        if isinstance(image_value, (bytes, bytearray)):
            return Image.open(BytesIO(image_value)).convert("RGB")

        raise TypeError(f"Unsupported image type: {type(image_value)}")

    def __iter__(self):
        """
        Iterate over parquet rows.

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

        if self.split in {"train", "train+val"}:
            generator = torch.Generator()
            generator.manual_seed(self.SHUFFLE_SEED + self.epoch)
            row_indices = torch.randperm(
                self.num_examples,
                generator=generator,
            ).tolist()
        else:
            row_indices = range(self.num_examples)

        for example_idx, row_idx in enumerate(row_indices):
            if example_idx % num_global_workers != global_worker_id:
                continue

            row_dict = self.dataframe.iloc[row_idx].to_dict()

            img = self._decode_image(row_dict["image"])
            label = int(row_dict["label"])

            if not (0 <= label < self.N_CLASSES):
                raise ValueError(
                    f"Invalid DMLab label={label}. "
                    f"Expected label in [0, {self.N_CLASSES - 1}]."
                )

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
