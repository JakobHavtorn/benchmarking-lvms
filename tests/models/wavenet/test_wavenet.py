"""
Test WaveNet model
"""

import numpy as np
import torch

import pytest

from blvm.models.wavenet import WaveNet, InputSizeError
from blvm.modules.distributions import CategoricalDense, GaussianDense


N_LAYERS = 5  # 10 in paper
N_STACKS = 2  # 5 in paper
IN_CHANNELS = 1  # 256 in paper. quantized and one-hot input.
RES_CHANNELS = 512  # 512 in paper
LIKELIHOOD = GaussianDense(RES_CHANNELS, IN_CHANNELS)


def generate_dummy(dummy_length):
    # x = np.arange(0, dummy_length, dtype=np.float32)
    x = np.random.uniform(low=-1, high=1, size=(1, dummy_length)).astype(np.float32)
    x = np.reshape(x, [1, dummy_length, IN_CHANNELS or 1])  # [B, T, C]
    x = torch.from_numpy(x)
    x_sl = torch.LongTensor([dummy_length])
    return x, x_sl


@pytest.fixture
def wavenet():
    net = WaveNet(
        n_layers=N_LAYERS,
        n_stacks=N_STACKS,
        in_channels=IN_CHANNELS,
        res_channels=RES_CHANNELS,
        likelihood=LIKELIHOOD,
    )
    return net


def test_wavenet_output_size(wavenet):
    """Input and output sizes must be the same"""
    # input size = receptive field size + 1
    # output size = receptive field size + 1
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)
    loss, metrics, output = wavenet(x, x_sl, pad_receptive_field=True)
    assert output.predictions.shape == torch.Size([1, wavenet.receptive_field + 1, IN_CHANNELS])

    # input size = receptive field size + 1
    # output size = 1
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)
    loss, metrics, output = wavenet(x, x_sl, pad_receptive_field=False)
    assert output.predictions.shape == torch.Size([1, 1, IN_CHANNELS])

    # input size = receptive field size + 1
    # output size = 1 (due to skip_size)
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)
    loss, metrics, output = wavenet(x, x_sl, pad_receptive_field=False, pad_causal=False)
    assert output.predictions.shape == torch.Size([1, 1, IN_CHANNELS])


def test_wavenet_fail_with_short_input(wavenet):
    """Wavenet must fail when given too short input"""
    x, x_sl = generate_dummy(wavenet.receptive_field)

    with pytest.raises(InputSizeError):
        loss, metrics, output = wavenet(x, x_sl, pad_receptive_field=False)  # Should fail. Input size is too short.


def test_wavenet_causality_gradient_full(wavenet):
    """Test causality using gradient computed on full output"""
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)  # (1, 65, 1)

    x.requires_grad_(True)
    x.retain_grad()

    loss, metrics, output = wavenet(x, x_sl)

    loss.backward()

    assert (x.grad[:, :-1, :] != 0).all(), "Gradient of loss wrt. full input is nonzero everywhere except last timestep"
    assert x.grad[:, -1, :] == 0, "Gradient of loss wrt. full input can never reach last timestep"


@pytest.mark.parametrize("slice_idx", [1, 5, 30, 64, 65])
def test_wavenet_causality_gradient_slice(wavenet, slice_idx):
    """Test causality using gradient computed on sliced output"""
    x, x_sl = generate_dummy(wavenet.receptive_field + 1)  # (1, 65, 1)

    x.requires_grad_(True)
    x.retain_grad()

    loss, metrics, output = wavenet(x, x_sl)
    output.log_prob_twise[:, :slice_idx].sum().backward()

    assert (
        x.grad[:, : slice_idx - 1, :] != 0
    ).all(), "Gradient of loss wrt. sliced input must be nonzero before sliced timestep"
    assert (
        x.grad[:, slice_idx:, :] == 0
    ).all(), "Gradient of loss wrt. sliced input must be zero after sliced timestep"
