import pytest
import torch
import torch.nn.functional as F

from .adaptformer import ViTB16AdaptFormer
from ..backbone import ViTB16Backbone


IMG_SIZE = 224
N_CLASSES = 10
BOTTLENECK_DIM = 64
ADAPTER_SCALE = 0.1
ADAPTER_DROPOUT = 0.1


@pytest.fixture(scope="module")
def backbone():
    model = ViTB16Backbone(img_size=IMG_SIZE)
    model.freeze()
    model.eval()
    return model


@pytest.fixture
def model(backbone):
    return ViTB16AdaptFormer(
        backbone=backbone,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
        adapter_scale=ADAPTER_SCALE,
        adapter_dropout=ADAPTER_DROPOUT,
    )


def test_forward_features_shapes(model):
    model.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        features = model.forward_features(x)

    assert set(features) == {"cls", "patch_tokens"}
    assert features["cls"].shape == (2, model.backbone.embed_dim)
    assert features["patch_tokens"].shape == (
        2,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


def test_forward_output_shape(model):
    model.eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (2, N_CLASSES)


def test_adapter_parameter_shapes(model):
    assert len(model.adapters) == model.backbone.num_layers

    for adapter in model.adapters:
        assert adapter.down.weight.shape == (BOTTLENECK_DIM, model.backbone.embed_dim)
        assert adapter.down.bias.shape == (BOTTLENECK_DIM,)
        assert adapter.up.weight.shape == (model.backbone.embed_dim, BOTTLENECK_DIM)
        assert adapter.up.bias.shape == (model.backbone.embed_dim,)


def test_adapter_initialization(model):
    for adapter in model.adapters:
        assert torch.count_nonzero(adapter.down.weight) > 0
        assert torch.count_nonzero(adapter.down.bias) == 0
        assert torch.count_nonzero(adapter.up.weight) == 0
        assert torch.count_nonzero(adapter.up.bias) == 0


def test_zero_bottleneck_dim(backbone):
    model = ViTB16AdaptFormer(
        backbone=backbone,
        n_classes=N_CLASSES,
        bottleneck_dim=0,
    )
    model.eval()
    assert len(model.adapters) == 0

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        features = model.forward_features(x)
        expected = backbone.forward_features(x)

    assert torch.equal(features["cls"], expected["cls"])
    assert torch.equal(features["patch_tokens"], expected["patch_tokens"])


def test_adapter_receives_gradients(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (2,))

    F.cross_entropy(model(x), target).backward()

    for adapter in model.adapters:
        for parameter in adapter.parameters():
            assert parameter.grad is not None
            assert parameter.grad.shape == parameter.shape


def test_frozen_backbone_receives_no_gradients(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (2,))

    F.cross_entropy(model(x), target).backward()

    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_train_does_not_change_backbone_mode(model):
    backbone = model.backbone
    assert not backbone.training

    model.train()
    assert model.training
    assert model.head.training
    assert model.adapters.training
    assert all(adapter.training for adapter in model.adapters)
    assert not backbone.training

    model.eval()
    assert not model.training
    assert not model.head.training
    assert not model.adapters.training
    assert all(not adapter.training for adapter in model.adapters)
    assert not backbone.training


def test_invalid_adapter_configuration_raises(backbone):
    with pytest.raises(ValueError, match="bottleneck_dim must be non-negative"):
        ViTB16AdaptFormer(
            backbone=backbone,
            n_classes=N_CLASSES,
            bottleneck_dim=-1,
        )

    with pytest.raises(ValueError, match="adapter_scale must be positive"):
        ViTB16AdaptFormer(
            backbone=backbone,
            n_classes=N_CLASSES,
            bottleneck_dim=BOTTLENECK_DIM,
            adapter_scale=0.0,
        )

    with pytest.raises(ValueError, match="adapter_dropout must be in"):
        ViTB16AdaptFormer(
            backbone=backbone,
            n_classes=N_CLASSES,
            bottleneck_dim=BOTTLENECK_DIM,
            adapter_dropout=1.0,
        )


def test_compute_loss(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, N_CLASSES, (2,))

    logits, loss, components = model.compute_loss(x, y)

    assert logits.shape == (2, N_CLASSES)
    assert logits.grad_fn is not None
    assert loss.shape == ()
    assert loss.grad_fn is not None
    assert "cross_entropy_loss" in components

    model.eval()
    with torch.no_grad():
        logits, loss, components = model.compute_loss(x, y)

    assert components["cross_entropy_loss"] == pytest.approx(
        F.cross_entropy(logits, y).item()
    )


def test_save_and_load_trainable_parameters(backbone, tmp_path):
    source = ViTB16AdaptFormer(backbone, N_CLASSES, BOTTLENECK_DIM)
    target = ViTB16AdaptFormer(backbone, N_CLASSES, BOTTLENECK_DIM)

    with torch.no_grad():
        for source_adapter in source.adapters:
            source_adapter.down.weight.fill_(1.23)
            source_adapter.down.bias.fill_(2.34)
            source_adapter.up.weight.fill_(3.45)
            source_adapter.up.bias.fill_(4.56)

        source.head.weight.fill_(5.67)
        source.head.bias.fill_(7.89)

    path = tmp_path / "adaptformer_trainable.pt"
    source.save(path)

    state = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {"head.weight", "head.bias"}

    for layer_idx in range(backbone.num_layers):
        expected_keys.update(
            {
                f"adapters.{layer_idx}.down.weight",
                f"adapters.{layer_idx}.down.bias",
                f"adapters.{layer_idx}.up.weight",
                f"adapters.{layer_idx}.up.bias",
            }
        )

    assert set(state) == expected_keys

    target.load_from_path(path)

    for source_adapter, target_adapter in zip(source.adapters, target.adapters):
        for name in ("weight", "bias"):
            assert torch.equal(
                getattr(target_adapter.down, name),
                getattr(source_adapter.down, name),
            )
            assert torch.equal(
                getattr(target_adapter.up, name),
                getattr(source_adapter.up, name),
            )

    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.head.bias, source.head.bias)
