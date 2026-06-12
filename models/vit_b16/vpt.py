import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from .backbone import ViTB16Backbone


class ViTB16VPT(nn.Module):
    """
    VPT model implemented for a ViT-B/16 backbone.
    """

    def __init__(
        self,
        img_size: int,
        n_classes: int,
        prompt_length: int = 0,
        prompt_mode: str = "shallow",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.n_classes = n_classes
        self.prompt_length = prompt_length
        self.prompt_mode = prompt_mode
        self.is_backbone_frozen = freeze_backbone

        # Backbone
        self.backbone = ViTB16Backbone(img_size=img_size)
        if freeze_backbone:
            self.freeze_backbone()

        # Head
        self.head = nn.Linear(self.backbone.embed_dim, n_classes)

        # Prompt
        self.prompt = nn.Parameter(
            torch.zeros(
                len(self.backbone.blocks) if prompt_mode == "deep" else 1,
                self.prompt_length,
                self.backbone.embed_dim,
            )
        )
        self.prompt_dropout_layer = nn.Dropout(0.0 if prompt_mode == "shallow" else 0.1)
        self._init_prompt()

    def freeze_backbone(self):
        self.backbone.freeze()
        self.backbone.eval()
        self.is_backbone_frozen = True

    def unfreeze_backbone(self):
        self.backbone.unfreeze()
        self.backbone.train(self.training)
        self.is_backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.is_backbone_frozen:
            self.backbone.eval()

        return self

    def eval(self):
        return self.train(False)

    def _init_prompt(self):
        if self.prompt_length == 0:
            return

        patch_size = (self.backbone.patch_size, self.backbone.patch_size)
        prompt_dim = self.backbone.embed_dim
        val = math.sqrt(6.0 / float(3 * reduce(mul, patch_size, 1) + prompt_dim))
        nn.init.uniform_(self.prompt.data, -val, val)

    def _get_prompt_tokens(self, batch_size: int, layer_idx: int = 0) -> torch.Tensor:
        prompt = self.prompt[layer_idx if self.prompt_mode == "deep" else 0]
        prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return self.prompt_dropout_layer(prompt)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer blocks + final norm with VPT prompt injection.

        Input tokens must contain only:
        [CLS, PATCH_1, ..., PATCH_N]

        Output tokens contain:
        [CLS, PROMPT_1, ..., PROMPT_P, PATCH_1, ..., PATCH_N]
        """
        h = tokens
        batch_size = h.shape[0]

        if self.prompt_length == 0:
            return self.backbone.encode_tokens(h)

        if self.prompt_mode == "shallow":
            prompt = self._get_prompt_tokens(batch_size)
            h = torch.cat([h[:, :1], prompt, h[:, 1:]], dim=1)

            for blk in self.backbone.blocks:
                h, _ = blk(h)

            return self.backbone.norm(h)

        if self.prompt_mode == "deep":
            for layer_idx, blk in enumerate(self.backbone.blocks):
                prompt = self._get_prompt_tokens(batch_size, layer_idx)

                if layer_idx == 0:
                    h = torch.cat([h[:, :1], prompt, h[:, 1:]], dim=1)
                else:
                    h = torch.cat(
                        [h[:, :1], prompt, h[:, 1 + self.prompt_length :]],
                        dim=1,
                    )

                h, _ = blk(h)

            return self.backbone.norm(h)

        raise ValueError(f"Invalid prompt_mode: {self.prompt_mode}")

    def forward_features(self, x: torch.Tensor):
        """
        Standard VPT forward.
        Returns final CLS and patch tokens.
        """
        batch_size = x.shape[0]

        patch_tokens = self.backbone.get_patch_tokens(x)
        cls_token = self.backbone.get_cls_token(batch_size)
        cls_pos, patch_pos = self.backbone.get_positional_embeddings()

        tokens = torch.cat(
            [cls_token + cls_pos, patch_tokens + patch_pos],
            dim=1,
        )
        tokens = self.backbone.input_dropout(tokens)
        encoded = self.encode_tokens(tokens)

        prompt_offset = self.prompt_length if self.prompt_length > 0 else 0

        return {
            "cls": encoded[:, 0],
            "prompt_tokens": encoded[:, 1 : 1 + prompt_offset],
            "patch_tokens": encoded[:, 1 + prompt_offset :],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features["cls"])

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)

        loss_components = {
            "cross_entropy_loss": loss.item(),
        }

        return logits, loss, loss_components

    def get_param_count(self):
        total_count = sum(p.numel() for p in self.parameters())
        backbone_count = sum(p.numel() for p in self.backbone.parameters())
        trainable_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_count": total_count,
            "backbone_count": backbone_count,
            "trainable_count": trainable_count,
            "trainable_over_total": (
                trainable_count / total_count if total_count > 0 else 0.0
            ),
            "trainable_over_backbone": (
                trainable_count / backbone_count if backbone_count > 0 else 0.0
            ),
        }

    def save(self, path: str):
        """
        Save only VPT trainable parameters from model state_dict:
        prompt tokens and classification head.
        """
        trainable_state_dict = {
            key: value.detach().cpu()
            for key, value in self.state_dict().items()
            if key == "prompt" or key.startswith("head.")
        }

        torch.save(trainable_state_dict, path)

    def load_from_path(self, path: str, map_location="cpu"):
        """
        Load only prompt tokens and classification head.
        """
        trainable_state_dict = torch.load(path, map_location=map_location)

        self.load_state_dict(trainable_state_dict, strict=False)
