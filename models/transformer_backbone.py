from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
import torch.nn as nn


class TransformerBackbone(nn.Module, ABC):
    """
    Minimal backbone interface required by the shared PEFT framework.

    The interface makes no assumptions about constructor arguments, internal
    module names, or the concrete Transformer implementation.

    The current functional definitions require:
        - one distinguished CLS representation;
        - spatial patch tokens with fixed coordinates;
        - a homogeneous Transformer stack;
        - multi-head attention with a shared output projection;
        - a two-layer MLP with a shared output projection.

    Concrete PEFT methods may require additional backbone capabilities beyond
    this shared interface.
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Residual-stream / CLS representation dimension."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of Transformer layers."""
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Number of attention heads per layer."""
        ...

    @property
    @abstractmethod
    def head_dim(self) -> int:
        """Dimension of one attention head."""
        ...

    @property
    @abstractmethod
    def mlp_dim(self) -> int:
        """Hidden dimension of the pretrained Transformer MLP."""
        ...

    @abstractmethod
    def get_patch_coordinates(self) -> torch.Tensor:
        """
        Return patch coordinates in the same order as patch tokens.

        Shape:
            [P, K]

        P is the number of patch tokens. Only Euclidean distances between
        coordinates are used, so K need not be two.
        """
        ...

    @abstractmethod
    def get_attention_output_weight(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Return the attention output-projection weight for one layer.

        Shape:
            [embed_dim, num_heads * head_dim]
        """
        ...

    @abstractmethod
    def get_mlp_output_weight(
        self,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Return the pretrained MLP output-projection weight for one layer.

        Shape:
            [embed_dim, mlp_dim]
        """
        ...

    @abstractmethod
    def forward_features(
        self,
        x: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """
        Run the native backbone forward.

        The returned mapping must contain at least:
            "cls": [B, embed_dim]
        """
        ...

    @abstractmethod
    def get_layerwise_features(
        self,
        x: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        """
        Return the intermediate representations after every Transformer layer.

        The returned mapping must contain at least:
            "cls": [L, B, embed_dim]
        """
        ...

    @abstractmethod
    def freeze(self) -> None:
        """Freeze all native backbone parameters."""
        ...

    @abstractmethod
    def unfreeze(self) -> None:
        """Unfreeze all native backbone parameters."""
        ...
