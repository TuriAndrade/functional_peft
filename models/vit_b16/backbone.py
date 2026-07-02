import os

import numpy as np
import torch
import torch.nn as nn

from third_party.vit_pytorch.models.modeling import VisionTransformer
from third_party.vit_pytorch.models.configs import get_b16_config


class ViTB16Backbone(nn.Module):
    """
    Loads the original Google ViT-B/16 ImageNet-21k checkpoint
    into a compatible PyTorch implementation.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()

        config = get_b16_config()
        vit = VisionTransformer(
            config,
            img_size=img_size,
            num_classes=21843,  # matches the default in the implementation
            zero_head=False,
            vis=False,
        )

        _CKPT_PATH = os.path.join("third_party", "vit_pytorch", "ckpts", "ViT-B_16.npz")
        weights = np.load(_CKPT_PATH)
        vit.load_from(weights)

        # Remove classification head so this acts like a backbone
        vit.head = nn.Identity()

        self.vit = vit
        self.embed_dim = config.hidden_size
        self.img_size = img_size
        self.patch_size = 16
        self.grid_size = img_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size

    @property
    def embeddings(self):
        return self.vit.transformer.embeddings

    @property
    def blocks(self):
        return self.vit.transformer.encoder.layer

    @property
    def norm(self):
        return self.vit.transformer.encoder.encoder_norm

    def freeze(self):
        for p in self.vit.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.vit.parameters():
            p.requires_grad = True

    def get_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns patch tokens before CLS and before positional embeddings.
        Shape: [B, 196, 768] for 224x224 input.
        """
        x = self.embeddings.patch_embeddings(x)  # [B, 768, 14, 14]
        x = x.flatten(2).transpose(-1, -2)  # [B, 196, 768]
        return x

    def get_cls_token(self, batch_size: int) -> torch.Tensor:
        """
        Returns the CLS token expanded to batch size.
        Shape: [B, 1, 768]
        """
        return self.embeddings.cls_token.expand(batch_size, -1, -1)

    def get_positional_embeddings(self):
        """
        Returns:
          cls_pos:   [1, 1, 768]
          patch_pos: [1, 196, 768]
        """
        pos = self.embeddings.position_embeddings
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        return cls_pos, patch_pos

    def input_dropout(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embeddings.dropout(tokens)

    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Runs transformer blocks + final norm on a token sequence.
        """
        h = tokens
        for blk in self.blocks:
            h, _ = blk(h)
        h = self.norm(h)
        return h

    def forward_features(self, x: torch.Tensor):
        """
        Standard backbone forward.
        Returns final CLS and patch tokens.
        """
        B = x.shape[0]
        patch_tokens = self.get_patch_tokens(x)
        cls_token = self.get_cls_token(B)
        cls_pos, patch_pos = self.get_positional_embeddings()

        tokens = torch.cat(
            [cls_token + cls_pos, patch_tokens + patch_pos],
            dim=1,
        )
        tokens = self.input_dropout(tokens)
        encoded = self.encode_tokens(tokens)

        return {
            "cls": encoded[:, 0],  # [B, 768]
            "patch_tokens": encoded[:, 1:],  # [B, 196, 768]
        }

    @staticmethod
    def _g2(p):
        if p is None:
            return None
        return torch.zeros_like(p) if p.grad is None else p.grad.detach().square()

    @staticmethod
    def _mean(*xs):
        xs = [x for x in xs if x is not None]
        return sum(x.sum() for x in xs) / sum(x.numel() for x in xs)

    @staticmethod
    def _group_units(x, n_groups=64):
        # x: [n_blocks, n_units, n_roles]
        n_groups = min(n_groups, x.shape[1])
        if x.shape[1] % n_groups == 0:
            return x.reshape(
                x.shape[0], n_groups, x.shape[1] // n_groups, x.shape[2]
            ).mean(2)
        return torch.stack(
            [c.mean(1) for c in torch.tensor_split(x, n_groups, dim=1)], 1
        )

    def _input_fim(self):
        patch = self.embeddings.patch_embeddings
        w = self._g2(patch.weight)

        patch_num = w.flatten(1).sum(1)
        patch_den = w[0].numel()

        if patch.bias is not None:
            patch_num = patch_num + self._g2(patch.bias)
            patch_den += 1

        pos = self._g2(self.embeddings.position_embeddings)
        pos = pos[:, 1 : 1 + self.num_patches].mean((0, 2))

        return patch_num / patch_den, pos.reshape(self.grid_size, self.grid_size)

    def _block_fim(self, block):
        attn, ffn = block.attn, block.ffn
        H, Dh = attn.num_attention_heads, attn.attention_head_size
        D, M = self.embed_dim, ffn.fc1.weight.shape[0]

        qw, kw, vw, ow = [
            self._g2(m.weight) for m in [attn.query, attn.key, attn.value, attn.out]
        ]
        qb, kb, vb, ob = [
            self._g2(m.bias) if m.bias is not None else None
            for m in [attn.query, attn.key, attn.value, attn.out]
        ]
        fc1w, fc2w = self._g2(ffn.fc1.weight), self._g2(ffn.fc2.weight)
        fc1b = self._g2(ffn.fc1.bias) if ffn.fc1.bias is not None else None
        fc2b = self._g2(ffn.fc2.bias) if ffn.fc2.bias is not None else None

        q = qw.reshape(H, Dh, D)
        k = kw.reshape(H, Dh, D)
        v = vw.reshape(H, Dh, D)
        o = ow.t().reshape(H, Dh, D)

        qk_num, qk_den = q.sum(-1) + k.sum(-1), 2 * D
        if qb is not None:
            qk_num, qk_den = qk_num + qb.reshape(H, Dh), qk_den + 1
        if kb is not None:
            qk_num, qk_den = qk_num + kb.reshape(H, Dh), qk_den + 1

        ov_num, ov_den = v.sum(-1) + o.sum(-1), 2 * D
        if vb is not None:
            ov_num, ov_den = ov_num + vb.reshape(H, Dh), ov_den + 1

        attention_circuit_dim = torch.stack([qk_num / qk_den, ov_num / ov_den], 1)

        mlp_key_num, mlp_key_den = fc1w.sum(1), D
        if fc1b is not None:
            mlp_key_num, mlp_key_den = mlp_key_num + fc1b, mlp_key_den + 1

        mlp_memory_unit = torch.stack([mlp_key_num / mlp_key_den, fc2w.mean(0)], -1)

        read = (qw.sum(0) + kw.sum(0) + vw.sum(0) + fc1w.sum(0)) / (3 * D + M)

        write_num, write_den = ow.sum(1) + fc2w.sum(1), D + M
        if ob is not None:
            write_num, write_den = write_num + ob, write_den + 1
        if fc2b is not None:
            write_num, write_den = write_num + fc2b, write_den + 1

        residual_io_channel = torch.stack([read, write_num / write_den])

        block_path = torch.stack(
            [
                self._mean(
                    self._g2(block.attention_norm.weight),
                    self._g2(block.attention_norm.bias),
                ),
                self._mean(qw, qb, kw, kb),
                self._mean(vw, vb, ow, ob),
                self._mean(
                    self._g2(block.ffn_norm.weight), self._g2(block.ffn_norm.bias)
                ),
                self._mean(fc1w, fc1b),
                self._mean(fc2w, fc2b),
            ]
        )

        return block_path, attention_circuit_dim, mlp_memory_unit, residual_io_channel

    @torch.no_grad()
    def aggregate_fim_grads(self, n_mlp_groups: int = 64):
        """
        Returned FIM groups:

        input_patch_channel: [D]
            FIM per patch-embedding output channel.
            For ViT-B/16: [768].

        input_pos_grid: [G, G]
            FIM per learned patch positional embedding, arranged as the image patch grid.
            For 224x224 ViT-B/16 with 16x16 patches: [14, 14].

        block_path: [B, 6]
            FIM per transformer block and computational path.
            Path axis: [LN1, QK, OV, LN2, MLP key, MLP value].
            For ViT-B/16: [12, 6].

        attention_circuit_head: [B, H, 2]
            FIM per block, attention head, and attention circuit.
            Circuit axis: [QK, OV].
            For ViT-B/16: [12, 12, 2].

        mlp_memory_group: [B, K, 2]
            FIM per block, grouped MLP hidden units, and MLP memory role.
            Role axis: [key, value].
            K defaults to 64 groups.
            For ViT-B/16: [12, 64, 2].

        residual_io_channel: [B, 2, D]
            FIM per block, residual-stream role, and residual channel.
            Role axis: [read, write].
            For ViT-B/16: [12, 2, 768].

        attention_circuit_dim: [B, H, 2, Dh]
            Fine-grained attention FIM per block, head, circuit, and head dimension.
            Circuit axis: [QK, OV].
            For ViT-B/16: [12, 12, 2, 64].

        mlp_memory_unit: [B, M, 2]
            Fine-grained MLP FIM per block, MLP hidden unit, and memory role.
            Role axis: [key, value].
            For ViT-B/16: [12, 3072, 2].
        """

        input_patch_channel, input_pos_grid = self._input_fim()

        block_path, attn_dim, mlp_unit, res_io = zip(
            *(self._block_fim(block) for block in self.blocks)
        )

        block_path = torch.stack(block_path)
        attn_dim = torch.stack(attn_dim)
        mlp_unit = torch.stack(mlp_unit)
        res_io = torch.stack(res_io)

        return {
            "input_patch_channel": input_patch_channel,
            "input_pos_grid": input_pos_grid,
            "block_path": block_path,
            "attention_circuit_head": attn_dim.mean(-1),
            "mlp_memory_group": self._group_units(mlp_unit, n_mlp_groups),
            "residual_io_channel": res_io,
            "attention_circuit_dim": attn_dim,
            "mlp_memory_unit": mlp_unit,
        }
