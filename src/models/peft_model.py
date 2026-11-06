"""Common PEFT model interface."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_backbone import TransformerBackbone


class PEFTModel(nn.Module, ABC):
    """Common PEFT model interface."""

    def __init__(
        self,
        backbone: TransformerBackbone,
        n_classes: int,
    ):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        super().__init__()
        self.backbone = backbone
        self.n_classes = int(n_classes)
        self.head = nn.Linear(self.backbone.embed_dim, n_classes)
        self.init_head()

    def train(self, mode: bool = True):
        self.training = mode
        for name, module in self.named_children():
            if name != "backbone":
                module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def init_head(self, init_from: nn.Linear | None = None) -> None:
        if init_from is None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
            return
        if (
            init_from.weight.shape != self.head.weight.shape
            or init_from.bias is None
            or init_from.bias.shape != self.head.bias.shape
        ):
            raise ValueError(
                f"Head shape mismatch: expected weight={tuple(self.head.weight.shape)}, "
                f"bias={tuple(self.head.bias.shape)}."
            )
        with torch.no_grad():
            self.head.weight.copy_(init_from.weight)
            self.head.bias.copy_(init_from.bias)

    @abstractmethod
    def init_peft(self, init_from: Any = None) -> None: ...

    def forward_features(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)["cls"])

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return logits, loss, {"cross_entropy_loss": loss.item()}

    def get_param_count(self):
        params = list(self.parameters())
        total = sum(p.numel() for p in params)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        trainable = sum(p.numel() for p in params if p.requires_grad)
        return {
            "total_count": total,
            "backbone_count": backbone,
            "trainable_count": trainable,
            "trainable_over_total": trainable / total if total else 0.0,
            "trainable_over_backbone": trainable / backbone if backbone else 0.0,
        }

    @abstractmethod
    def save(self, path: str) -> None: ...

    def load_from_path(self, path: str, map_location="cpu") -> None:
        self.load_state_dict(
            torch.load(path, map_location=map_location),
            strict=False,
        )
