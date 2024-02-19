"""Modules for WaveNet"""

import math

from typing import Optional

import torch
import torch.nn as nn

from blvm.modules.activations import GatedTanhUnit
from blvm.utils.convolutions import compute_conv_attributes


class CausalConv1d(nn.Module):
    """Causal Convolution for WaveNet. Causality imposed by removing last timestep of output (and left same padding).

    The output of this convolution is `y = causal_conv(x)` where `y[t]` depends on `x[:t]` (doesn't include `x[t]`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        activation: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        self.activation = activation() if activation else None

    def init_weights_for_test(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.fill_(1)
                if hasattr(m, "bias"):
                    m.bias.data.fill_(0)

    def causal_padding(self, x: torch.Tensor):
        """Remove last input (causal convolution)"""
        return nn.functional.pad(x, (0, -1))

    def forward(self, x: torch.Tensor, pad_causal: bool = True):
        if pad_causal:
            x = self.causal_padding(x)
        output = self.conv(x)
        if self.activation is not None:
            output = self.activation(output)
        return output


class Conv1dResidualGLU(nn.Module):
    def __init__(
        self,
        res_channels: int,
        skip_channels: Optional[int] = None,
        gate_channels: Optional[int] = None,
        kernel_size: int = 2,
        dilation: int = 1,
        bias: bool = True,
        activation: nn.Module = GatedTanhUnit,
    ):
        """Residual 1d convolution with gated linear unit (GLU)

        Args:
            res_channels (int): number of channels of input (x) and output (o).
            skip_channels (int): number of channels of skip connection (s). Defaults to `res_channels`.
            gate_channels (int): number of channels of gated activation units (g). Defaults to `2 * res_channels`.
            kernel_size (int): kernel size of convolutions
            dilation (int): amount of dilation
        """
        super().__init__()
        if skip_channels is None:
            skip_channels = res_channels

        if gate_channels is None:
            gate_channels = 2 * res_channels

        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.gate_channels = gate_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

        self.inv_std = math.sqrt(0.5)

        self.conv = nn.Conv1d(res_channels, gate_channels, kernel_size=kernel_size, dilation=dilation)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1x1rs = nn.Conv1d(gate_out_channels, res_channels + skip_channels, kernel_size=1, bias=bias)

        self.activation = activation(dim=1)

    def forward(
        self, x: torch.Tensor, skip_size: int, c: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x (torch.Tensor): Input tensor (B, C, T)
            skip_size (int): The last output size for loss and prediction
        """
        pre_act = self.conv(x)

        act = self.activation(pre_act)

        rs = self.conv1x1rs(act)
        r, s = rs[:, :self.res_channels], rs[:, self.res_channels:]

        s = s[..., -skip_size:]  # Ignore padding
        x = x[:, :, -r.size(2) :]  # Remove whatever the gated_dilated kernel ate

        o = (r + x) * self.inv_std
        # print(f"resblock: mean={o.mean().item():.2f}, std={o.std().item():.2f}")
        return o, s


class ResidualStack(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_stacks: int,
        res_channels: int,
        skip_channels: Optional[int] = None,
        gate_channels: Optional[int] = None,
        kernel_size: int = 2,
        base_dilation: int = 2,
        in_channels: int = None,
        activation: nn.Module = GatedTanhUnit,
    ):
        """Stack residual blocks by layer and stack size

        Args:
            n_layers (int): Number of stacked residual blocks (k). Dilations chosen as 1, 2, 4, 8, 16, 32, 64...
            n_stacks (int): Number of stacks of residual blocks with skip connections to the output.
            res_channels (int): Number of channels in the residual connections.
            skip_channels (int): number of channels of skip connection (s). Defaults to `res_channels`.
            gate_channels (int): number of channels of gated activation units (g). Defaults to `2 * res_channels`.
            in_channels (int): Special number of input channels if different from `res_channels`.
        """
        super().__init__()

        in_channels = res_channels if in_channels is None else in_channels

        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.gate_channels = gate_channels
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.in_channels = in_channels

        self.dilations = self.build_dilations(n_layers, n_stacks, base_dilation)

        self.receptive_fields = self.compute_receptive_field(n_layers, n_stacks, kernel_size, base_dilation)
        self.receptive_field = self.receptive_fields[-1]

        if self.in_channels is not None:
            self.in_transform = nn.Conv1d(in_channels, res_channels, kernel_size=1)

        res_blocks = nn.ModuleList()
        for l, dilation in enumerate(self.dilations):
            block = Conv1dResidualGLU(
                res_channels=res_channels,
                skip_channels=skip_channels,
                gate_channels=gate_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                activation=activation,
            )
            res_blocks.append(block)

        self.res_blocks = res_blocks

    @staticmethod
    def build_dilations(n_layers: int, n_stacks: int, base_dilation: int):
        """Return a list of dilations {1, 2, 4, 8, 16, ...} with a dilation for each of the residual blocks"""
        if base_dilation > 1:
            return [1, *[base_dilation * 2 ** i for i in range(0, n_layers - 1)]] * n_stacks
        return [1] * n_layers * n_stacks

    @staticmethod
    def compute_receptive_field(n_layers: int, n_stacks: int, kernel_size: int, base_dilation: int):
        """Compute and return the receptive field at every block of the ResidualStack"""
        kernels = [kernel_size] * n_layers * n_stacks
        paddings = [0] * n_layers * n_stacks
        strides = [1] * n_layers * n_stacks
        dilations = [1, *[base_dilation * 2 ** i for i in range(0, n_layers - 1)]] * n_stacks
        o, s, r, c = compute_conv_attributes(kernels, paddings, strides, dilations, return_all=True)
        return r

    def forward(self, x: torch.Tensor, skip_size: int, c: Optional[torch.Tensor] = None, g: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Input of shape (B, Cr, T)
            c (torch.Tensor): Local conditioning auxiliary features of shape (B, Cc, T)
            g (torch.Tensor): Global conditioning auxiliary features of shape (B, Cg, T)
            skip_size (int): The last output size for loss and prediction
        """
        o = x if self.in_channels is None else self.in_transform(x)
        skips = []

        for i, res_block in enumerate(self.res_blocks):
            # o is the next input
            o, s = res_block(o, skip_size, c, g)
            skips.append(s)

        return skips


class PointwiseTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: nn.Module = nn.ReLU):
        r"""The last network of WaveNet. Outputs logits for the likelihood module.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output classes

        Shapes:
            x (torch.Tensor): Input of shape :math:`(B, C, T)` where :math:`C=\text{in\_channels}`
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.act1 = activation()
        self.linear = nn.Linear(in_channels, out_channels)
        self.act2 = activation()

    def forward(self, x: torch.Tensor):
        output = x.transpose(1, 2)  # (B, C, T) to (B, T, C)
        output = self.act1(output)
        output = self.linear(output)
        output = self.act2(output)
        return output  # (B, T, C)
