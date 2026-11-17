import torch

from shop_core.temporal.tcn import ActionTCN, Checkpoint, TCNConfig

CLASSES = ("a", "b", "c", "d", "e")


def _model(temporal: bool = True) -> ActionTCN:
    torch.manual_seed(0)
    return ActionTCN(TCNConfig(in_channels=12, classes=CLASSES, hidden=16, temporal=temporal)).eval()


def test_output_shape_matches_input_length():
    out = _model()(torch.randn(2, 12, 37))
    assert out.shape == (2, 5, 37)


def test_tcn_is_causal_future_steps_do_not_change_past_outputs():
    model = _model()
    x = torch.randn(1, 12, 60)
    with torch.inference_mode():
        base = model(x)
        y = x.clone()
        y[:, :, 40:] = torch.randn(1, 12, 20) * 50
        changed = model(y)
    assert torch.allclose(base[:, :, :40], changed[:, :, :40], atol=1e-6)
    assert not torch.allclose(base[:, :, 40:], changed[:, :, 40:])


def test_receptive_field_is_bounded():
    cfg = TCNConfig(in_channels=12, classes=CLASSES, hidden=16)
    model = _model()
    x = torch.randn(1, 12, 80)
    with torch.inference_mode():
        base = model(x)[:, :, -1]
        y = x.clone()
        y[:, :, : 80 - cfg.receptive_field] += 100.0
        assert torch.allclose(base, model(y)[:, :, -1], atol=1e-5)


def test_linear_baseline_has_no_temporal_context():
    model = _model(temporal=False)
    assert model.config.receptive_field == 1
    x = torch.randn(1, 12, 10)
    with torch.inference_mode():
        y = x.clone()
        y[:, :, 3] += 10
        diff = (model(x) - model(y)).abs().sum(dim=1)[0]
    assert diff[3] > 0 and float(diff.sum() - diff[3]) == 0.0


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    ckpt = Checkpoint(model.config, model.state_dict(), torch.zeros(12), torch.ones(12), [0.5] * 5, {"run": "t"})
    ckpt.save(tmp_path / "m.pt")
    loaded = Checkpoint.load(tmp_path / "m.pt")
    x = torch.randn(1, 12, 20)
    with torch.inference_mode():
        assert torch.allclose(model(x), loaded.build()(x))
    assert loaded.config.classes == CLASSES and loaded.metadata["run"] == "t"
