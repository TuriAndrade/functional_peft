from types import MethodType

import torch
import torch.nn as nn

from ..peft_model import PEFTModel
from .backbone import ViTB16Backbone


class ViTB16SSF(PEFTModel):
    """Scale-and-Shift Fine-Tuning for ViT-B/16."""

    _ALL_SSF_TARGETS = (
        "patch",
        "attention_norm",
        "query",
        "key",
        "value",
        "attention_out",
        "ffn_norm",
        "fc1",
        "fc2",
        "final_norm",
    )

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        freeze_backbone: bool = True,
        ssf_targets=(
            "patch",
            "attention_norm",
            "query",
            "key",
            "value",
            "attention_out",
            "ffn_norm",
            "fc1",
            "fc2",
            "final_norm",
        ),
    ):
        ssf_targets = tuple(ssf_targets)

        if len(set(ssf_targets)) != len(ssf_targets):
            raise ValueError("ssf_targets must not contain duplicates")

        if any(name not in self._ALL_SSF_TARGETS for name in ssf_targets):
            raise ValueError(
                "ssf_targets may contain only "
                + ", ".join(repr(name) for name in self._ALL_SSF_TARGETS)
            )

        super().__init__(
            backbone=ViTB16Backbone(img_size=img_size),
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
        )

        self.img_size = img_size
        self.ssf_targets = ssf_targets

        self.ssf_scale = nn.ParameterDict()
        self.ssf_shift = nn.ParameterDict()

        shapes = {
            "patch": (self.backbone.embed_dim,),
            "attention_norm": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "query": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "key": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "value": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "attention_out": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "ffn_norm": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "fc1": (
                self.backbone.num_layers,
                self.backbone.mlp_dim,
            ),
            "fc2": (
                self.backbone.num_layers,
                self.backbone.embed_dim,
            ),
            "final_norm": (self.backbone.embed_dim,),
        }

        reference = next(self.backbone.parameters())

        for name in ssf_targets:
            self.ssf_scale[name] = nn.Parameter(reference.new_empty(shapes[name]))
            self.ssf_shift[name] = nn.Parameter(reference.new_empty(shapes[name]))

        self.init_peft()

        if ssf_targets:
            self._patch_ssf_targets()

    def init_peft(
        self,
        init_from: tuple[nn.ParameterDict, nn.ParameterDict] | None = None,
    ) -> None:
        if not self.ssf_targets:
            return

        if init_from is not None:
            scales, shifts = init_from

            if set(scales) != set(self.ssf_scale) or set(shifts) != set(self.ssf_shift):
                raise ValueError(
                    f"SSF target mismatch: expected {self.ssf_targets}, "
                    f"got scales={tuple(scales)}, shifts={tuple(shifts)}."
                )

            for name in self.ssf_targets:
                if (
                    scales[name].shape != self.ssf_scale[name].shape
                    or shifts[name].shape != self.ssf_shift[name].shape
                ):
                    raise ValueError(
                        f"SSF shape mismatch for target {name!r}: expected "
                        f"scale={tuple(self.ssf_scale[name].shape)}, "
                        f"shift={tuple(self.ssf_shift[name].shape)}, got "
                        f"scale={tuple(scales[name].shape)}, "
                        f"shift={tuple(shifts[name].shape)}."
                    )

            with torch.no_grad():
                for name in self.ssf_targets:
                    self.ssf_scale[name].copy_(scales[name])
                    self.ssf_shift[name].copy_(shifts[name])

            return

        with torch.no_grad():
            for scale in self.ssf_scale.values():
                nn.init.normal_(scale, mean=1.0, std=0.02)

            for shift in self.ssf_shift.values():
                nn.init.normal_(shift, mean=0.0, std=0.02)

    @staticmethod
    def _apply_affine(
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] == scale.shape[0]:
            return x * scale + shift

        shape = (
            1,
            -1,
            *(1 for _ in range(x.ndim - 2)),
        )

        return x * scale.view(shape) + shift.view(shape)

    def _apply_ssf(
        self,
        x: torch.Tensor,
        name: str,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        scale = self.ssf_scale[name]
        shift = self.ssf_shift[name]

        if layer_idx is not None:
            scale = scale[layer_idx]
            shift = shift[layer_idx]

        return self._apply_affine(
            x,
            scale,
            shift,
        )

    def _patch_ssf_targets(self) -> None:
        def patch(module, name, layer_idx=None):
            original_forward = module.forward

            def forward(_, x):
                return self._apply_ssf(
                    original_forward(x),
                    name,
                    layer_idx,
                )

            module.forward = MethodType(
                forward,
                module,
            )

        if "patch" in self.ssf_targets:
            patch(
                self.backbone.embeddings.patch_embeddings,
                "patch",
            )

        layer_targets = {
            "attention_norm": lambda block: block.attention_norm,
            "query": lambda block: block.attn.query,
            "key": lambda block: block.attn.key,
            "value": lambda block: block.attn.value,
            "attention_out": lambda block: block.attn.out,
            "ffn_norm": lambda block: block.ffn_norm,
            "fc1": lambda block: block.ffn.fc1,
            "fc2": lambda block: block.ffn.fc2,
        }

        for layer_idx, block in enumerate(self.backbone.blocks):
            for name in self.ssf_targets:
                if name in layer_targets:
                    patch(
                        layer_targets[name](block),
                        name,
                        layer_idx,
                    )

        if "final_norm" in self.ssf_targets:
            patch(
                self.backbone.norm,
                "final_norm",
            )

    def save(
        self,
        path: str,
    ) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if (
                key.startswith("head.")
                or key.startswith("ssf_scale.")
                or key.startswith("ssf_shift.")
                or (not self.is_backbone_frozen and key.startswith("backbone."))
            )
        }

        torch.save(
            state_dict,
            path,
        )
