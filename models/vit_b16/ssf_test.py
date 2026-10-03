import torch
import pytest
import torch.nn.functional as F

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


def test_forward_features_shapes():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        features = model.forward_features(x)

    expected_keys = {"cls", "patch_tokens"}
    assert set(features.keys()) == expected_keys

    assert features["cls"].shape == (batch_size, model.backbone.embed_dim)
    assert features["patch_tokens"].shape == (
        batch_size,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


def test_forward_output_shape():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, N_CLASSES)


def test_ssf_parameter_shapes():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )

    n_layers = len(model.backbone.blocks)
    embed_dim = model.backbone.embed_dim
    mlp_dim = model.backbone.blocks[0].ffn.fc1.out_features

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

    assert set(model.ssf_scale.keys()) == set(SSF_TARGETS)
    assert set(model.ssf_shift.keys()) == set(SSF_TARGETS)

    for name, shape in expected_shapes.items():
        assert model.ssf_scale[name].shape == shape
        assert model.ssf_shift[name].shape == shape


def test_ssf_initialization():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )

    for name in SSF_TARGETS:
        scale = model.ssf_scale[name]
        shift = model.ssf_shift[name]

        assert torch.isfinite(scale).all()
        assert torch.isfinite(shift).all()

        # Official SSF initialization:
        # scale ~ N(1, 0.02^2), shift ~ N(0, 0.02^2).
        assert torch.count_nonzero(scale - 1.0) > 0
        assert torch.count_nonzero(shift) > 0


def test_empty_ssf_targets():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=(),
        freeze_backbone=True,
    )
    model.eval()

    assert len(model.ssf_scale) == 0
    assert len(model.ssf_shift) == 0
    assert model.ssf_targets == ()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        features = model.forward_features(x)

    assert features["cls"].shape == (batch_size, model.backbone.embed_dim)
    assert features["patch_tokens"].shape == (
        batch_size,
        model.backbone.num_patches,
        model.backbone.embed_dim,
    )


def test_ssf_target_subset():
    targets = ("query", "value", "fc1")

    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=targets,
    )

    assert model.ssf_targets == targets
    assert set(model.ssf_scale.keys()) == set(targets)
    assert set(model.ssf_shift.keys()) == set(targets)


def test_ssf_receives_gradients():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )
    model.train()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (batch_size,))

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    for name in SSF_TARGETS:
        scale = model.ssf_scale[name]
        shift = model.ssf_shift[name]

        assert scale.grad is not None
        assert scale.grad.shape == scale.shape

        assert shift.grad is not None
        assert shift.grad.shape == shift.shape


def test_frozen_backbone_receives_no_gradients():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
        freeze_backbone=True,
    )
    model.train()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (batch_size,))

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    assert all(p.grad is None for p in model.backbone.parameters())
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_train_keeps_frozen_backbone_in_eval_mode():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
        freeze_backbone=True,
    )

    model.train()

    assert model.training
    assert model.head.training
    assert not model.backbone.training
    assert model.is_backbone_frozen
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_unfreeze_backbone_restores_train_mode():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
        freeze_backbone=True,
    )

    model.train()
    model.unfreeze_backbone()

    assert model.training
    assert model.backbone.training
    assert not model.is_backbone_frozen
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_freeze_backbone_sets_eval_mode_and_disables_gradients():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
        freeze_backbone=False,
    )

    model.train()
    model.freeze_backbone()

    assert model.training
    assert not model.backbone.training
    assert model.is_backbone_frozen
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_empty_targets_with_unfrozen_backbone_behaves_as_full_ft():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=(),
        freeze_backbone=False,
    )

    assert len(model.ssf_scale) == 0
    assert len(model.ssf_shift) == 0
    assert not model.is_backbone_frozen
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_invalid_ssf_target_raises():
    with pytest.raises(ValueError, match="ssf_targets may contain only"):
        ViTB16SSF(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            ssf_targets=("invalid",),
        )


def test_duplicate_ssf_target_raises():
    with pytest.raises(ValueError, match="ssf_targets must not contain duplicates"):
        ViTB16SSF(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            ssf_targets=("query", "query"),
        )


def test_compute_loss():
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )
    model.train()

    batch_size = 4
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    y = torch.randint(0, N_CLASSES, (batch_size,))

    logits, loss, loss_components = model.compute_loss(x, y)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, N_CLASSES)
    assert logits.grad_fn is not None

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()
    assert loss.grad_fn is not None

    assert isinstance(loss_components, dict)
    assert "cross_entropy_loss" in loss_components

    model.eval()
    with torch.no_grad():
        logits, loss, loss_components = model.compute_loss(x, y)
        expected_loss = F.cross_entropy(logits, y).item()

    assert pytest.approx(loss_components["cross_entropy_loss"]) == loss.item()
    assert pytest.approx(loss_components["cross_entropy_loss"]) == expected_loss


def test_save_and_load_trainable_parameters(tmp_path):
    source_model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )

    target_model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )

    # Force known values so the test is deterministic.
    with torch.no_grad():
        for name in SSF_TARGETS:
            source_model.ssf_scale[name].fill_(1.23)
            source_model.ssf_shift[name].fill_(2.34)

            target_model.ssf_scale[name].fill_(0.0)
            target_model.ssf_shift[name].fill_(0.0)

        source_model.head.weight.fill_(4.56)
        source_model.head.bias.fill_(7.89)

        target_model.head.weight.fill_(0.0)
        target_model.head.bias.fill_(0.0)

    checkpoint_path = tmp_path / "ssf_trainable.pt"

    source_model.save(checkpoint_path)

    saved_state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    expected_keys = {
        "head.weight",
        "head.bias",
    }

    expected_keys.update(f"ssf_scale.{name}" for name in SSF_TARGETS)
    expected_keys.update(f"ssf_shift.{name}" for name in SSF_TARGETS)

    assert set(saved_state_dict.keys()) == expected_keys

    target_model.load_from_path(checkpoint_path)

    for name in SSF_TARGETS:
        assert torch.allclose(
            target_model.ssf_scale[name],
            source_model.ssf_scale[name],
        )

        assert torch.allclose(
            target_model.ssf_shift[name],
            source_model.ssf_shift[name],
        )

    assert torch.allclose(
        target_model.head.weight,
        source_model.head.weight,
    )

    assert torch.allclose(
        target_model.head.bias,
        source_model.head.bias,
    )


def test_functional_gates_preserve_forward(tmp_path):
    model = ViTB16SSF(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        ssf_targets=SSF_TARGETS,
    )
    model.eval()

    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        expected = model.forward_features(x)

    param_count_before = model.get_param_count()

    model.prepare_functional_analysis(
        tmp_path / "spectral_basis.pt",
    )
    model.reset_functional_gates()

    assert set(model.functional_gates) == {
        "attention_qk",
        "attention_ov",
        "mlp_in",
        "mlp_out",
    }

    assert model.functional_gates["attention_qk"].shape == (
        model.backbone.num_layers,
        model.backbone.num_heads,
        model.backbone.head_dim,
    )
    assert model.functional_gates["attention_ov"].shape == (
        model.backbone.num_layers,
        model.backbone.num_heads,
        model.backbone.head_dim,
    )
    assert model.functional_gates["mlp_in"].shape == (
        model.backbone.num_layers,
        model.backbone.embed_dim,
    )
    assert model.functional_gates["mlp_out"].shape == (
        model.backbone.num_layers,
        model.backbone.embed_dim,
    )

    assert all(torch.all(gate == 1) for gate in model.functional_gates.values())
    assert all(not gate.requires_grad for gate in model.functional_gates.values())

    # Functional gates must not affect PEFT parameter counting.
    assert model.get_param_count() == param_count_before

    with torch.no_grad():
        actual = model.forward_features(x)

    assert torch.allclose(
        actual["cls"],
        expected["cls"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert torch.allclose(
        actual["patch_tokens"],
        expected["patch_tokens"],
        rtol=1e-5,
        atol=1e-6,
    )
