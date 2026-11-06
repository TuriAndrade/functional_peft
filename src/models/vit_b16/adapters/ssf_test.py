import pytest
import torch
import torch.nn.functional as F

from ..backbone import ViTB16Backbone
from .ssf import ViTB16SSF


IMG_SIZE = 224
N_CLASSES = 10

SSF_TARGETS = (
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


@pytest.fixture(scope="module")
def backbone():
    model = ViTB16Backbone(img_size=IMG_SIZE)
    model.freeze()
    model.eval()
    return model


@pytest.fixture
def model(backbone):
    return ViTB16SSF(
        backbone=backbone,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
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


def test_ssf_parameter_shapes(model):
    n_layers = model.backbone.num_layers
    embed_dim = model.backbone.embed_dim
    mlp_dim = model.backbone.mlp_dim

    expected_shapes = {
        "patch": (embed_dim,),
        "attention_norm": (n_layers, embed_dim),
        "query": (n_layers, embed_dim),
        "key": (n_layers, embed_dim),
        "value": (n_layers, embed_dim),
        "attention_out": (n_layers, embed_dim),
        "ffn_norm": (n_layers, embed_dim),
        "fc1": (n_layers, mlp_dim),
        "fc2": (n_layers, embed_dim),
        "final_norm": (embed_dim,),
    }

    assert set(model.ssf_scale) == set(SSF_TARGETS)
    assert set(model.ssf_shift) == set(SSF_TARGETS)

    for name, shape in expected_shapes.items():
        assert model.ssf_scale[name].shape == shape
        assert model.ssf_shift[name].shape == shape


def test_ssf_initialization(model):
    for name in SSF_TARGETS:
        scale = model.ssf_scale[name]
        shift = model.ssf_shift[name]

        assert torch.isfinite(scale).all()
        assert torch.isfinite(shift).all()
        assert torch.count_nonzero(scale - 1.0) > 0
        assert torch.count_nonzero(shift) > 0


def test_empty_ssf_targets(backbone):
    model = ViTB16SSF(
        backbone=backbone,
        n_classes=N_CLASSES,
        ssf_targets=(),
    )
    model.eval()

    assert len(model.ssf_scale) == 0
    assert len(model.ssf_shift) == 0
    assert model.ssf_targets == ()

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        features = model.forward_features(x)
        expected = backbone.forward_features(x)

    assert torch.equal(features["cls"], expected["cls"])
    assert torch.equal(features["patch_tokens"], expected["patch_tokens"])


def test_ssf_target_subset(backbone):
    targets = ("query", "value", "fc1")
    model = ViTB16SSF(
        backbone=backbone,
        n_classes=N_CLASSES,
        ssf_targets=targets,
    )

    assert model.ssf_targets == targets
    assert set(model.ssf_scale) == set(targets)
    assert set(model.ssf_shift) == set(targets)


def test_ssf_receives_gradients(model):
    model.train()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (2,))

    F.cross_entropy(model(x), target).backward()

    for name in SSF_TARGETS:
        scale = model.ssf_scale[name]
        shift = model.ssf_shift[name]

        assert scale.grad is not None
        assert scale.grad.shape == scale.shape
        assert shift.grad is not None
        assert shift.grad.shape == shift.shape


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


def test_invalid_ssf_target_raises(backbone):
    with pytest.raises(ValueError, match="ssf_targets may contain only"):
        ViTB16SSF(
            backbone=backbone,
            n_classes=N_CLASSES,
            ssf_targets=("invalid",),
        )


def test_duplicate_ssf_target_raises(backbone):
    with pytest.raises(ValueError, match="ssf_targets must not contain duplicates"):
        ViTB16SSF(
            backbone=backbone,
            n_classes=N_CLASSES,
            ssf_targets=("query", "query"),
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
    source = ViTB16SSF(
        backbone=backbone,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )
    target = ViTB16SSF(
        backbone=backbone,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )

    with torch.no_grad():
        for name in SSF_TARGETS:
            source.ssf_scale[name].fill_(1.23)
            source.ssf_shift[name].fill_(2.34)

        source.head.weight.fill_(4.56)
        source.head.bias.fill_(7.89)

    path = tmp_path / "ssf_trainable.pt"
    source.save(path)

    state = torch.load(path, map_location="cpu", weights_only=True)
    expected_keys = {"head.weight", "head.bias"}
    expected_keys.update(f"ssf_scale.{name}" for name in SSF_TARGETS)
    expected_keys.update(f"ssf_shift.{name}" for name in SSF_TARGETS)

    assert set(state) == expected_keys

    target.load_from_path(path)

    for name in SSF_TARGETS:
        assert torch.equal(target.ssf_scale[name], source.ssf_scale[name])
        assert torch.equal(target.ssf_shift[name], source.ssf_shift[name])

    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.head.bias, source.head.bias)
