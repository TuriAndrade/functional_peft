import pytest
import torch
import torch.nn.functional as F

from ..backbone import ViTB16Backbone
from .lora import ViTB16LoRA


IMG_SIZE = 224
N_CLASSES = 10
RANK = 8
LORA_TARGETS = ("query", "value")


@pytest.fixture(scope="module")
def backbone():
    model = ViTB16Backbone(img_size=IMG_SIZE)
    model.freeze()
    model.eval()
    return model


@pytest.fixture
def model(backbone):
    return ViTB16LoRA(
        backbone=backbone,
        n_classes=N_CLASSES,
        rank=RANK,
        lora_targets=LORA_TARGETS,
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


def test_lora_parameter_shapes(model):
    assert model.lora_A.shape == (
        model.backbone.num_layers,
        len(LORA_TARGETS),
        RANK,
        model.backbone.embed_dim,
    )
    assert model.lora_B.shape == (
        model.backbone.num_layers,
        len(LORA_TARGETS),
        model.backbone.embed_dim,
        RANK,
    )


def test_lora_initialization(model):
    assert torch.count_nonzero(model.lora_A) > 0
    assert torch.count_nonzero(model.lora_B) == 0


def test_zero_rank(backbone):
    model = ViTB16LoRA(backbone=backbone, n_classes=N_CLASSES, rank=0)
    model.eval()

    assert model.lora_A is None
    assert model.lora_B is None

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        features = model.forward_features(x)
        expected = backbone.forward_features(x)

    assert torch.equal(features["cls"], expected["cls"])
    assert torch.equal(features["patch_tokens"], expected["patch_tokens"])


def test_lora_receives_gradients(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (2,))

    loss = F.cross_entropy(model(x), target)
    loss.backward()

    assert model.lora_A.grad is not None
    assert model.lora_A.grad.shape == model.lora_A.shape
    assert model.lora_B.grad is not None
    assert model.lora_B.grad.shape == model.lora_B.shape


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
    assert not backbone.training

    model.eval()
    assert not model.training
    assert not model.head.training
    assert not backbone.training


def test_shared_backbone_isolation(backbone):
    first = ViTB16LoRA(backbone, N_CLASSES, rank=RANK)
    second = ViTB16LoRA(backbone, N_CLASSES, rank=RANK)
    first.eval()
    second.eval()

    assert first.backbone is second.backbone

    with torch.no_grad():
        first.lora_B.normal_(std=0.01)
        second.lora_B.normal_(std=0.01)

    x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        before = first.forward_features(x)["cls"].clone()
        second.forward_features(x)
        after = first.forward_features(x)["cls"]

    assert torch.equal(before, after)


def test_invalid_lora_target_raises(backbone):
    with pytest.raises(ValueError, match="lora_targets may contain only"):
        ViTB16LoRA(
            backbone=backbone,
            n_classes=N_CLASSES,
            rank=RANK,
            lora_targets=("key",),
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
    source = ViTB16LoRA(backbone, N_CLASSES, rank=RANK)
    target = ViTB16LoRA(backbone, N_CLASSES, rank=RANK)

    with torch.no_grad():
        source.lora_A.fill_(1.23)
        source.lora_B.fill_(2.34)
        source.head.weight.fill_(4.56)
        source.head.bias.fill_(7.89)

    path = tmp_path / "lora_trainable.pt"
    source.save(path)

    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state) == {"lora_A", "lora_B", "head.weight", "head.bias"}

    target.load_from_path(path)

    for name in ("lora_A", "lora_B"):
        assert torch.equal(getattr(target, name), getattr(source, name))

    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.head.bias, source.head.bias)
