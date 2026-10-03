import torch
import pytest
import torch.nn.functional as F

from .adaptformer import ViTB16AdaptFormer

IMG_SIZE = 224
N_CLASSES = 10
BOTTLENECK_DIM = 64
ADAPTER_SCALE = 0.1
ADAPTER_DROPOUT = 0.1


def test_forward_features_shapes():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
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
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )
    model.eval()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, N_CLASSES)


def test_adapter_parameter_shapes():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )

    n_layers = len(model.backbone.blocks)
    embed_dim = model.backbone.embed_dim

    assert len(model.adapters) == n_layers

    for adapter in model.adapters:
        assert adapter.down.weight.shape == (
            BOTTLENECK_DIM,
            embed_dim,
        )
        assert adapter.down.bias.shape == (BOTTLENECK_DIM,)
        assert adapter.up.weight.shape == (
            embed_dim,
            BOTTLENECK_DIM,
        )
        assert adapter.up.bias.shape == (embed_dim,)


def test_adapter_initialization():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )

    for adapter in model.adapters:
        assert torch.count_nonzero(adapter.down.weight) > 0
        assert torch.count_nonzero(adapter.down.bias) == 0
        assert torch.count_nonzero(adapter.up.weight) == 0
        assert torch.count_nonzero(adapter.up.bias) == 0


def test_zero_bottleneck_dim():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=0,
    )
    model.eval()

    assert len(model.adapters) == 0

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


def test_adapter_receives_gradients():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )
    model.train()

    batch_size = 2
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE)
    target = torch.randint(0, N_CLASSES, (batch_size,))

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    for adapter in model.adapters:
        assert adapter.down.weight.grad is not None
        assert adapter.down.bias.grad is not None
        assert adapter.up.weight.grad is not None
        assert adapter.up.bias.grad is not None

        assert adapter.down.weight.grad.shape == adapter.down.weight.shape
        assert adapter.down.bias.grad.shape == adapter.down.bias.shape
        assert adapter.up.weight.grad.shape == adapter.up.weight.shape
        assert adapter.up.bias.grad.shape == adapter.up.bias.shape


def test_frozen_backbone_receives_no_gradients():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
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
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
        freeze_backbone=True,
    )

    model.train()

    assert model.training
    assert model.head.training
    assert model.adapters.training
    assert not model.backbone.training
    assert model.is_backbone_frozen
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_unfreeze_backbone_restores_train_mode():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
        freeze_backbone=True,
    )

    model.train()
    model.unfreeze_backbone()

    assert model.training
    assert model.backbone.training
    assert not model.is_backbone_frozen
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_freeze_backbone_sets_eval_mode_and_disables_gradients():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
        freeze_backbone=False,
    )

    model.train()
    model.freeze_backbone()

    assert model.training
    assert not model.backbone.training
    assert model.is_backbone_frozen
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_invalid_adapter_configuration_raises():
    with pytest.raises(ValueError, match="bottleneck_dim must be non-negative"):
        ViTB16AdaptFormer(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            bottleneck_dim=-1,
        )

    with pytest.raises(ValueError, match="adapter_scale must be positive"):
        ViTB16AdaptFormer(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            bottleneck_dim=BOTTLENECK_DIM,
            adapter_scale=0.0,
        )

    with pytest.raises(ValueError, match="adapter_dropout must be in"):
        ViTB16AdaptFormer(
            img_size=IMG_SIZE,
            n_classes=N_CLASSES,
            bottleneck_dim=BOTTLENECK_DIM,
            adapter_dropout=1.0,
        )


def test_compute_loss():
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
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
    source_model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )

    target_model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )

    # Force known values so the test is deterministic.
    with torch.no_grad():
        for source_adapter, target_adapter in zip(
            source_model.adapters,
            target_model.adapters,
        ):
            source_adapter.down.weight.fill_(1.23)
            source_adapter.down.bias.fill_(2.34)
            source_adapter.up.weight.fill_(3.45)
            source_adapter.up.bias.fill_(4.56)

            target_adapter.down.weight.zero_()
            target_adapter.down.bias.zero_()
            target_adapter.up.weight.zero_()
            target_adapter.up.bias.zero_()

        source_model.head.weight.fill_(5.67)
        source_model.head.bias.fill_(7.89)

        target_model.head.weight.zero_()
        target_model.head.bias.zero_()

    checkpoint_path = tmp_path / "adaptformer_trainable.pt"

    source_model.save(checkpoint_path)

    saved_state_dict = torch.load(checkpoint_path, map_location="cpu")

    expected_keys = {
        "head.weight",
        "head.bias",
    }

    for layer_idx in range(len(source_model.adapters)):
        expected_keys.update(
            {
                f"adapters.{layer_idx}.down.weight",
                f"adapters.{layer_idx}.down.bias",
                f"adapters.{layer_idx}.up.weight",
                f"adapters.{layer_idx}.up.bias",
            }
        )

    assert set(saved_state_dict.keys()) == expected_keys

    target_model.load_from_path(checkpoint_path)

    for source_adapter, target_adapter in zip(
        source_model.adapters,
        target_model.adapters,
    ):
        assert torch.allclose(
            target_adapter.down.weight,
            source_adapter.down.weight,
        )
        assert torch.allclose(
            target_adapter.down.bias,
            source_adapter.down.bias,
        )
        assert torch.allclose(
            target_adapter.up.weight,
            source_adapter.up.weight,
        )
        assert torch.allclose(
            target_adapter.up.bias,
            source_adapter.up.bias,
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
    model = ViTB16AdaptFormer(
        img_size=IMG_SIZE,
        n_classes=N_CLASSES,
        bottleneck_dim=BOTTLENECK_DIM,
    )
    model.eval()

    # The up projection is zero-initialized. Make the complete adapter
    # non-zero so this test actually verifies preservation of AdaptFormer.
    with torch.no_grad():
        for adapter in model.adapters:
            adapter.up.weight.normal_()
            adapter.up.bias.normal_()

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
