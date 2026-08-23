import torch
from src.models.mmnet import MMNet


def test_monotonic_clamp_zeros_out_negative_weights():
    model = MMNet(n_features=5, monotonic_feature_idx=[0, 2])
    with torch.no_grad():
        model.shared1.weight[:, 0] = -1.0
        model.shared1.weight[:, 2] = -0.5
        model.shared1.weight[:, 1] = -2.0  # non-monotonic feature, should NOT be clamped

    model.clamp_monotonic_weights()

    assert (model.shared1.weight[:, 0] >= 0).all()
    assert (model.shared1.weight[:, 2] >= 0).all()
    assert (model.shared1.weight[:, 1] < 0).all()  # untouched


def test_forward_output_shape_and_range():
    model = MMNet(n_features=6, monotonic_feature_idx=[])
    model.eval()
    x = torch.randn(4, 6)
    out = model(x)
    assert out.shape == (4, 3)
    assert (out >= 0).all() and (out <= 1).all()


def test_param_count_matches_manual_sum():
    model = MMNet(n_features=18, monotonic_feature_idx=[])
    manual_count = sum(p.numel() for p in model.parameters())
    assert model.param_count() == manual_count