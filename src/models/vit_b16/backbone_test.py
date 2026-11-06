import pytest
import torch

from ..transformer_backbone import TransformerBackbone
from .backbone import ViTB16Backbone


@pytest.fixture(scope="module")
def backbone():
    model = ViTB16Backbone(img_size=224)
    model.eval()
    return model


def test_backbone_interface(backbone):
    assert isinstance(backbone, TransformerBackbone)


def test_backbone_metadata(backbone):
    assert backbone.embed_dim == 768
    assert backbone.num_layers == 12
    assert backbone.num_heads == 12
    assert backbone.head_dim == 64
    assert backbone.mlp_dim == 3072
    assert backbone.patch_size == 16
    assert backbone.grid_size == 14
    assert backbone.num_patches == 196


def test_get_patch_tokens_shape(backbone):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        patch_tokens = backbone.get_patch_tokens(x)
    assert patch_tokens.shape == (2, 196, 768)


def test_get_cls_token_shape(backbone):
    assert backbone.get_cls_token(batch_size=2).shape == (2, 1, 768)


def test_positional_embeddings_shape(backbone):
    cls_pos, patch_pos = backbone.get_positional_embeddings()
    assert cls_pos.shape == (1, 1, 768)
    assert patch_pos.shape == (1, 196, 768)


def test_encode_tokens_shape(backbone):
    tokens = torch.randn(2, 197, 768)
    with torch.no_grad():
        encoded = backbone.encode_tokens(tokens)
    assert encoded.shape == (2, 197, 768)


def test_forward_features_shape(backbone):
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = backbone.forward_features(x)
    assert set(out) == {"cls", "patch_tokens"}
    assert out["cls"].shape == (2, 768)
    assert out["patch_tokens"].shape == (2, 196, 768)


def test_freeze_backbone(backbone):
    backbone.freeze()
    assert all(not p.requires_grad for p in backbone.parameters())


def test_unfreeze_backbone(backbone):
    backbone.freeze()
    backbone.unfreeze()
    assert all(p.requires_grad for p in backbone.parameters())
