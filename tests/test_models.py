import pytest
import torch

from tug_transfer.models import DenseCNNBiLSTM, TimeStepMLPHead, WindowCNNBiLSTM


def test_dense_model_shape():
    model = DenseCNNBiLSTM(in_channels=27, num_classes=5)
    x = torch.randn(2, 31, 27)
    lengths = torch.tensor([31, 19])
    mask = torch.arange(31)[None, :] < lengths[:, None]
    logits = model(x, mask=mask, lengths=lengths)
    assert logits.shape == (2, 31, 5)
    features = model.encode(x, mask=mask, lengths=lengths)
    assert features.shape == (2, 31, 128)
    assert torch.allclose(features[1, 19:], torch.zeros_like(features[1, 19:]))


def test_dense_model_is_invariant_to_extra_batch_padding_in_eval_mode():
    torch.manual_seed(7)
    model = DenseCNNBiLSTM(in_channels=27, num_classes=5, dropout=0.0).eval()
    short = torch.randn(1, 17, 27)
    short_mask = torch.ones(1, 17, dtype=torch.bool)
    with torch.no_grad():
        alone = model(short, mask=short_mask, lengths=torch.tensor([17]))

        long = torch.randn(1, 41, 27)
        padded = torch.zeros(2, 41, 27)
        padded[0, :17] = short[0]
        padded[1] = long[0]
        lengths = torch.tensor([17, 41])
        mask = torch.arange(41)[None, :] < lengths[:, None]
        batched = model(padded, mask=mask, lengths=lengths)

    assert torch.allclose(alone[0], batched[0, :17], atol=1e-6, rtol=1e-5)


def test_dense_model_rejects_noncontiguous_mask():
    model = DenseCNNBiLSTM(in_channels=27, num_classes=5)
    x = torch.randn(1, 5, 27)
    mask = torch.tensor([[True, True, False, True, False]])
    with pytest.raises(ValueError, match="contiguous prefixes"):
        model(x, mask=mask, lengths=torch.tensor([3]))


def test_dense_model_backward_with_variable_lengths():
    torch.manual_seed(11)
    model = DenseCNNBiLSTM(in_channels=27, num_classes=5, dropout=0.1)
    x = torch.randn(3, 23, 27)
    lengths = torch.tensor([23, 17, 9])
    mask = torch.arange(23)[None, :] < lengths[:, None]
    logits = model(x, mask=mask, lengths=lengths)
    loss = logits[mask].square().mean()
    loss.backward()
    assert model.lstm.weight_ih_l0.grad is not None
    assert torch.isfinite(model.lstm.weight_ih_l0.grad).all()


def test_window_model_shape():
    model = WindowCNNBiLSTM(in_channels=27, num_classes=5)
    x = torch.randn(4, 40, 27)
    assert model(x).shape == (4, 5)


def test_timestep_head_shape():
    head = TimeStepMLPHead(input_dim=128, hidden_dims=(32,), num_classes=3)
    assert head(torch.randn(2, 20, 128)).shape == (2, 20, 3)
