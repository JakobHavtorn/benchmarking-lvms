import functools

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from torchtyping import TensorType
from blvm.modules.convolutions import ConvDepthwiseSeparable1d, ConvTransposeDepthwiseSeparable1d

from blvm.utils.convolutions import compute_conv_attributes_single


class TemporalResidual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: TensorType["B", "C", "T"]):
        x_residual = self.module(x)
        if x_residual.shape[-1] == x.shape[-1]:
            return x_residual + x

        x_resampled = torch.nn.functional.interpolate(x, size=x_residual.shape[-1], mode="nearest")
        return x_residual + x_resampled


class BlockSeparable(nn.Module):
    def __init__(
        self,
        channels_bottleneck,
        kernel_size,
        stride,
        dilation,
        activation_cls: nn.Module,
        transposed,
        channels_factor: int = 4,
        bias: bool = False,
    ):
        """Block with depthwise separable convolutions"""
        super().__init__()

        sep_conv_obj = ConvTransposeDepthwiseSeparable1d if transposed else ConvDepthwiseSeparable1d

        channels_block = channels_factor * channels_bottleneck

        transform = nn.Sequential(
            nn.Conv1d(channels_bottleneck, channels_block, 1, bias=bias),
            activation_cls(),
            nn.GroupNorm(num_channels=channels_block, num_groups=channels_block),  # Channel-wise normalization
            sep_conv_obj(
                in_channels=channels_block,
                out_channels=channels_bottleneck,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                normalization=nn.GroupNorm(num_channels=channels_block, num_groups=channels_block),
                activation=activation_cls(),
            ),
        )

        self.block = TemporalResidual(module=transform)

    def forward(self, x):
        return self.block(x)


class BlockSimple(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        stride,
        dilation,
        activation_cls: nn.Module,
        transposed,
        bias: bool = False,
    ):
        super().__init__()

        conv_obj = nn.ConvTranspose1d if transposed else nn.Conv1d

        conv = conv_obj(channels, channels, kernel_size, stride=stride, dilation=dilation, bias=bias)
        norm = nn.GroupNorm(num_channels=channels, num_groups=channels)  # Channel-wise normalization
        nonl = activation_cls()

        self.block = TemporalResidual(nn.Sequential(conv, norm, nonl))

    def forward(self, x):
        return self.block(x)


class ConvCoder1d(nn.Module):
    def __init__(
        self,
        strides: List[int],
        channels: int = 128,
        kernel_size: Union[int, List[int]] = 5,
        stride_per_block: int = 2,
        dilation_factor: int = 1,
        num_blocks: int = 8,
        channels_in: Optional[Union[int, List[Union[None, int]]]] = None,
        channels_out: Optional[Union[int, List[Union[None, int]]]] = None,
        transposed: bool = False,
        block_type: str = "BlockSeparable",
        activation: nn.Module = nn.PReLU,
    ):
        """Convolutional coder with different options for blocks.

        Default arguments above correspond to the highest performing model in the paper (Table II)

        The dimension of the embedding at layer `l` is:
        - `channels_out[l]` if `channels_out[l] is not None`
        - `channels` if `channels_out[l] is None`

        Args:
            channels: Number of channels in bottleneck 1 × 1-conv block
            channels_block: Number of channels in convolutional blocks
            kernel_size: Kernel size in convolutional blocks
            num_blocks: Number of convolutional blocks in each level
            num_levels: int: Number of levels
            channels_in: Channels for optional `channels_in` to `channels` projection for inputs.
            channels_out: Channels for optional `channels` to `channels_out` projection for all embeddings.
        """
        super().__init__()

        if block_type not in ["BlockSeparable", "BlockSimple"]:
            raise ValueError(f"Unknown {block_type=}.")

        num_levels = len(strides)
        overall_strides = np.cumprod(strides)

        assert all(stride_per_block ** num_blocks >= s for s in strides), f"Not enough blocks per level for {strides=}"

        self.strides = strides
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_blocks = num_blocks
        self.transposed = transposed
        self.stride_per_block = stride_per_block
        self.block_type = block_type
        self.activation = activation

        self.num_levels = num_levels
        self.overall_strides = overall_strides
        self.overall_stride = self.overall_strides[-1]

        if channels_in is None:
            self.channels_in = [None] * num_levels
        elif isinstance(channels_in, int):
            self.channels_in = [channels_in] + [None] * (num_levels - 1)
        else:
            self.channels_in = channels_in

        if channels_out is None:
            self.channels_out = [None] * num_levels
        elif isinstance(channels_out, int):
            self.channels_out = [channels_out] * num_levels
        else:
            self.channels_out = channels_out

        self.e_size = [c if c is not None else self.channels for c in self.channels_out]

        # Blocks
        block_cls = globals().get(block_type)

        self.overall_receptive_fields = []
        self.receptive_fields = []

        self.levels = nn.ModuleList()
        self.out_projs = nn.ModuleDict()
        self.in_projs = nn.ModuleDict()

        overall_stride_in, overall_receptive_field_in = 1, 1
        for l in range(num_levels):
            # build blocks
            remaining_stride = self.strides[l] if l > 0 else self.strides[0]

            stride_in, receptive_field_in = 1, 1
            blocks = []
            for b in range(num_blocks):
                dilation = dilation_factor ** b

                if remaining_stride >= self.stride_per_block:
                    stride = self.stride_per_block
                    remaining_stride = remaining_stride // self.stride_per_block
                else:
                    if remaining_stride != 1:
                        raise ValueError(f"{remaining_stride=} is not 1 at {l=}, {b=}.")
                    stride = 1

                block = block_cls(
                    channels,
                    kernel_size,
                    stride,
                    dilation,
                    activation,
                    transposed,
                    bias=True,
                )
                blocks += [block]

                # keep track of overall and per block receptive field
                _, overall_stride_in, overall_receptive_field_in, _ = compute_conv_attributes_single(
                    i=1,
                    k=kernel_size,
                    p=0,
                    s=stride,
                    d=dilation,
                    s_in=overall_stride_in,
                    r_in=overall_receptive_field_in,
                )
                _, stride_in, receptive_field_in, _ = compute_conv_attributes_single(
                    i=1,
                    k=kernel_size,
                    p=0,
                    s=stride,
                    d=dilation,
                    s_in=stride_in,
                    r_in=receptive_field_in,
                )

            self.overall_receptive_fields.append(overall_receptive_field_in)
            self.receptive_fields.append(receptive_field_in)

            # Flip blocks to stride in mirrored order.
            # The blocks are required to be symmetric to flipping in terms of amount of consumed padding.
            # Usually this corresponds to doing only a single strided convolution (or pooling etc.) per block.
            if transposed:
                blocks = blocks[::-1]

            self.levels.append(nn.Sequential(*blocks))

            # build out projection [B, channels, T] -> [B, self.channels_out, T] (1x1 convolution)
            if self.channels_out[l] is not None:
                self.out_projs[str(l)] = nn.Sequential(
                    nn.Conv1d(channels, self.channels_out[l], 1), activation()
                )

            # build in projection [B, channels_in, T] -> [B, channels, T] (1x1 convolution)
            if self.channels_in[l] is not None:
                self.in_projs[str(l)] = nn.Sequential(
                    nn.Conv1d(self.channels_in[l], channels, 1), activation()
                )

        self.overall_receptive_field = self.overall_receptive_fields[-1]

    @property
    def device(self):
        return self.levels[0][0][0].weight.device

    def pad_level(self, hidden: TensorType["B", "C", "T"], pad_left: int, pad_right: int):
        """Pad input to ensure that encodings and decodings have the same number of timesteps per level.

        If `pad_left` and `pad_right` are both True, padding is distributed evenly between both sides with
        any odd amount put on the right side. Otherwise all padding is put on the chosen side.

        If the coder is transposed, the same padding is that of a transposed convolution which actually removes
        outputs. Hence, in this case, the padding is assumed to be done after the convolution on its output.

        Args:
            hidden (tensor): Representation at `level` to pad.
            level (int): Coding level.
            pad_left (bool): Put padding on the left side.
            pad_right (bool): Put padding on the right side.
        """
        if not pad_left and not pad_right:
            return hidden

        if self.transposed:
            pad_left = -pad_left
            pad_right = -pad_right

        return torch.nn.functional.pad(hidden, [pad_left, pad_right])

    def forward_level(
        self,
        hidden: TensorType["B", "C", "T"],
        level: int,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> Tuple[TensorType["B", "D", "T"], TensorType["B", "D", "T"]]:
        hidden = self.in_projs[str(level)](hidden) if str(level) in self.in_projs else hidden
        if not self.transposed:
            hidden = self.pad_level(hidden, pad_left, pad_right)
        hidden = self.levels[level](hidden)
        if self.transposed:
            hidden = self.pad_level(hidden, pad_left, pad_right)
        encoding = self.out_projs[str(level)](hidden) if str(level) in self.out_projs else hidden
        return hidden, encoding

    def forward(
        self, hidden: TensorType["B", "C", "T"], pad_left: List[int] = None, pad_right: List[int] = None
    ) -> List[TensorType["B", "D", "T"]]:
        if pad_left is None:
            pad_left = [0] * self.num_levels
        if pad_right is None:
            pad_right = [0] * self.num_levels
        encodings = []
        for level in range(self.num_levels):
            hidden, encoding = self.forward_level(hidden, level, pad_left[level], pad_right[level])
            encodings.append(encoding)
        return encodings

    def __getitem__(self, level: int):
        return functools.partial(self.forward_level, level=level)

    def __len__(self):
        return self.num_levels
