from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
import torch.nn as nn


class TransformerBackbone(nn.Module, ABC):
    """Common Transformer backbone interface."""

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

    @property
    @abstractmethod
    def head_dim(self) -> int: ...

    @property
    @abstractmethod
    def mlp_dim(self) -> int: ...

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return at least `cls: [B, embed_dim]`."""
        ...

    @abstractmethod
    def freeze(self) -> None: ...

    @abstractmethod
    def unfreeze(self) -> None: ...
