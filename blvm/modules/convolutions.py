from torch import nn

from blvm.utils.convolutions import _single


class ConvDepthwiseSeparable1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        normalization: nn.Module = None,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.stride = _single(stride)
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.activation = activation
        self.transposed = False

        # [B, channels_block, T] -> [B, channels_block, T]
        self.depthwise_conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
        )
        self.activation = activation
        self.norm = normalization
        # [B, channels_block, T] -> [B, channels_bottleneck, T]
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, channels_block, T]
        Returns:
            result: [B, channels_bottleneck, T]
        """
        x = self.depthwise_conv(x)
        x = self.activation(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.pointwise_conv(x)
        return x


class ConvTransposeDepthwiseSeparable1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        normalization: nn.Module = None,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.stride = _single(stride)
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.transposed = True

        # [B, in_channels, T] -> [B, in_channels, (T_new] where T_new = T - kernel_size) // stride + 1
        self.depthwise_conv = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=True,
        )
        self.activation = activation
        self.norm = normalization
        # [B, in_channels, T_new] -> [B, out_channels, T_new]
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [B, channels_block, T]
        Returns:
            result: [B, channels_bottleneck, T]
        """
        x = self.depthwise_conv(x)
        x = self.activation(x)
        x = self.norm(x) if self.norm is not None else x
        x = self.pointwise_conv(x)
        return x
