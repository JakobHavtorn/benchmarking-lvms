"""WaveNet main model"""

import math

from types import SimpleNamespace
from typing import Optional, Union

import tqdm
import torch
import torch.nn as nn

from torchtyping import TensorType

from blvm.evaluation.metrics import BitsPerDimMetric, LLMetric, LossMetric
from blvm.utils.operations import sequence_mask, split_sequence, stack_tensor, unstack_tensor
from blvm.utils.padding import get_modulo_length, pad_to_length

from .wavenet_modules import CausalConv1d, PointwiseTransform, ResidualStack
from ..base_model import BaseModel


class InputSizeError(Exception):
    def __init__(self, input_size, receptive_field):
        message = "Input size has to be larger than receptive_field\n"
        message += f"Input size: {input_size}, Receptive fields size: {receptive_field}"
        super().__init__(message)


class WaveNet(BaseModel):
    def __init__(
        self,
        likelihood: nn.Module,
        in_channels: int = 1,
        embedding_dim: int = None,
        num_bins: int = 256,
        n_layers: int = 10,
        n_stacks: int = 5,
        res_channels: int = 512,
        skip_channels: Optional[int] = None,
        gate_channels: Optional[int] = None,
        kernel_size: int = 2,
        base_dilation: int = 2,
        n_stack_frames: int = 1,
        activation: nn.Module = nn.ReLU,
    ):
        """Stochastic autoregressive modelling of audio waveform frames with dilated causal convolutions [1].

        The total number of residual blocks (layers) used is equal to the n_layers times the n_stacks.
        This is `k` in Figure 4 in the paper.

        An illustration of the model:

                             |---------------------------------------------------| *residual*
                             |                                                   |
                             |             |-- tanh --|                          |
                 -> *input* ---> dilate ---|          * ---> 1x1 ---|----------- + ---> *input*
                                           |-- sigm --|             |
                                                                    |
                                                                    |
                 -> *skip* ---------------------------------------- + ----------------> *skip*

        Args:
            likelihood (nn.Module): Module for computing parameters of the output distribution and the loglikelihood.
            in_channels (int): Number of embeddings to (optionally) use for input. If 1, convolve the input without embedding.
            num_bins (int): Number of classes for output (i.e. number of quantized values in y audio values)
            n_layers (int): Number of stacked residual blocks. Dilations chosen as 2, 4, 8, 16, 32, 64...
            n_stacks (int): Number of stacks of residual blocks with skip connections to the output.
            res_channels (int): Number of channels in residual blocks (and embedding if num_embeddings > 1).
            g_use_embedding (bool): If True, assume g is integer that can be embedded. Otherwise use g directly.

        Reference:
            [1] WaveNet: A Generative Model for Raw Audio (https://arxiv.org/abs/1609.03499)
        """
        super().__init__()

        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.res_channels = res_channels
        self.skip_channels = skip_channels
        self.gate_channels = gate_channels
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.num_bins = num_bins
        self.n_stack_frames = n_stack_frames
        self.activation = activation

        if embedding_dim is not None and n_stack_frames > 1:
            raise ValueError("Cannot stack frames if using an embedding (which is what we do when in_channels>1)")

        if embedding_dim is not None and in_channels > 1:
            raise ValueError("Cannot use more than 1 input_channel if also wanting to use an embedding.")

        self.variance_scale = math.sqrt(1 / self.n_stacks * self.n_layers)

        if embedding_dim is None:
            self.embedding = None
            self.causal = CausalConv1d(
                in_channels=in_channels * n_stack_frames,
                out_channels=res_channels,
                kernel_size=kernel_size,
                activation=None,
            )
        else:
            self.embedding = nn.Embedding(num_embeddings=num_bins, embedding_dim=embedding_dim)
            self.causal = CausalConv1d(
                in_channels=embedding_dim,
                out_channels=res_channels,
                kernel_size=kernel_size,
                activation=None,
                groups=res_channels,
            )

        self.res_stack = ResidualStack(
            n_layers=n_layers,
            n_stacks=n_stacks,
            res_channels=res_channels,
            kernel_size=kernel_size,
            base_dilation=base_dilation,
        )
        self.receptive_field = self.res_stack.receptive_field + self.causal.kernel_size - 1

        self.out_transform = PointwiseTransform(res_channels, res_channels * n_stack_frames)

        self.likelihood = likelihood

    def compute_loss(
        self,
        y: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        parameters: torch.Tensor,
    ):
        """Compute the loss as negative log-likelihood per frame, masked and mormalized according to sequence lengths.

        Args:
            y (torch.LongTensor): Input audio waveform, i.e. the y (B, T).
            x_sl (torch.LongTensor): Sequence lengths of examples in the batch.
            parameters (torch.FloatTensor): Parameters for output distribution (likelihood).
        """
        seq_mask = sequence_mask(x_sl, max_len=y.size(1), device=y.device)  # (B, T)
        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask  # (B, T)
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        loss = -log_prob.nansum() / x_sl.nansum()  # sum B, normalize by sequence lengths
        return loss, log_prob, log_prob_twise

    def forward(
        self,
        x: Union[TensorType["B", "T"], TensorType["B", "T", "C"]],
        x_sl: TensorType["B", int],
        y: Union[TensorType["B", "T"], TensorType["B", "T", "C"]] = None,
        pad_causal: bool = True,
        pad_receptive_field: bool = True,
    ):
        """Reconstruct an input and compute the log-likelihood loss.

        The duration of x has to be longer than the receptive field if `pad_receptive_field` is `False`.

        Args:
            x (torch.Tensor): Audio waveform (batch, timestep, channels) with values in [-1, 1] (optinally dequantized)
            x_sl (torch.LongTensor): Sequence lengths of each example in the batch.
            y (torch.Tensor): An optional target y. Must have shape matching x after convolving (consumes RF).
            pad_causal (bool): Shifts input one frame to the left compared to y by removing the right-most input.
            pad_receptive_field (bool): If True, pads receptive field to left of input to predict on all of input.
                                        If False, does not predict on first receptive field number of inputs.
        """
        if y is None:
            y = x.detach()
            if not pad_receptive_field:
                # Remove receptive field from x (as y) if not padding since we condition on these values.
                y = y[:, self.receptive_field * self.n_stack_frames :]

        x_sl_strided = (x_sl / self.n_stack_frames).ceil().int()
        if self.n_stack_frames > 1:
            x, p = stack_tensor(x, self.n_stack_frames, dim=1)

        if self.embedding is None:
            x = x.unsqueeze(-1) if x.ndim == 2 else x  # (B, T) -> (B, T, 1)
            y = y.unsqueeze(-1) if y.ndim == 2 else y
        else:
            x = self.embedding(x)  # (B, T, C)

        x = x.transpose(1, 2)  # (B, C, T)

        if pad_receptive_field:
            skip_size = x.size(2)
            x = nn.functional.pad(x, (self.receptive_field, 0))
        else:
            skip_size = x.size(2) - self.receptive_field
            x_sl = x_sl - self.receptive_field

        if x.size(2) - int(pad_causal) < self.receptive_field:
            raise InputSizeError(x.size(2), self.receptive_field)

        output = self.causal(x, pad_causal=pad_causal)  # (B, C, T - causal kernel + 1 - pad_causal)
        skip_connections = self.res_stack(output, skip_size)  # (S, B, C, T - causal kernel + 1 - pad_causal)
        output = sum(skip_connections) * self.variance_scale  # (B, C, T)
        logits = self.out_transform(output)

        if self.n_stack_frames > 1:
            logits = unstack_tensor(logits, self.n_stack_frames, p, dim=-1)

        parameters = self.likelihood(logits)
        predictions = self.likelihood.sample(parameters)
        predictions_mode = self.likelihood.mode(parameters)

        loss, log_prob, log_prob_twise = self.compute_loss(y, x_sl, parameters)

        metrics = [
            LossMetric(loss, weight_by=log_prob.numel()),
            LLMetric(log_prob),
            BitsPerDimMetric(log_prob, reduce_by=x_sl),
        ]
        z = [s.permute(0, 2, 1) for s in skip_connections][::5]
        # [B, T, D] for each latent layer
        output = SimpleNamespace(
            loss=loss,
            log_prob=log_prob,
            log_prob_twise=log_prob_twise,
            parameters=parameters,
            z=z,  # (T, D)
            z_sl=x_sl_strided,
            y=y,
            predictions=predictions,
            predictions_mode=predictions_mode,
        )
        return loss, metrics, output

    def split_sequence(self, x: TensorType["B", "T", "C", float], x_sl: TensorType["B", int], length: int):
        """Split a long sequence into smaller subsequences for memory constrained `.forward()`.

        Overlap is `receptive_field` and not `receptive_field - 1` (rf - stride) because we apply `pad_causal`in
        forward which removes the last input for each subsequence which is then never seen.
        """
        overlap = self.receptive_field * self.n_stack_frames
        length = get_modulo_length(length, stride=self.n_stack_frames)
        mode = "extend" if overlap >= length else "consume"
        splits_x, splits_x_sl = split_sequence(x, x_sl, length=length, overlap=overlap, mode=mode)
        if mode == "extend":
            splits_x = [pad_to_length(split_x, overlap + length, "left", dim=1) for split_x in splits_x]
        return splits_x, splits_x_sl

    def forward_split(
        self,
        x: TensorType["B", "T", "C", float],
        x_sl: TensorType["B", int],
        i_split: int,
        y: TensorType["B", "T"] = None,
    ):
        """A convience forward method for split input sequences"""
        return self.forward(x, x_sl, y=y, pad_causal=True, pad_receptive_field=(i_split == 0))

    def generate(self, n_samples: int, n_frames: int = 48000, x: Optional[TensorType["B", "T", "C"]] = None):
        """Generate samples from the WaveNet starting from a zero vector

        TODO Improve this with a fast cached sampling method such as https://arxiv.org/pdf/1611.09482.pdf.
        """
        if x is None:
            if self.embedding is None:
                # start with floats of zeros
                x = torch.zeros(n_samples, self.receptive_field, self.in_channels * self.n_stack_frames, device=self.device)  # (B, T, C)
            else:
                # start with embeddings of the zeros
                x = torch.zeros(n_samples, self.receptive_field, device=self.device, dtype=torch.int64)  # (B, T)
                x = self.embedding(x)  # (B, T, C)

        x = x.transpose(1, 2)  # (B, C, T)

        x_hat = []
        for _ in tqdm.tqdm(range(n_frames)):
            output = self.causal(x, pad_causal=False)
            skip_connections = self.res_stack(output, skip_size=1)
            output = sum(skip_connections) / self.variance_scale  # (B, C, T)
            logits = self.out_transform(output)
            
            if self.n_stack_frames > 1:
                logits = unstack_tensor(logits, self.n_stack_frames, dim=-1)

            parameters = self.likelihood(logits)
            predictions = self.likelihood.sample(parameters)

            x_hat.append(predictions)

            # prepare predictions as next input
            if self.embedding is not None:
                predictions = self.embedding(predictions)  # (B, T, C) (1, 1, C)

            # FIFO along T since we don't need to consider inputs beyond the receptive field
            x = torch.cat([x[:, :, 1:], predictions], dim=2)

        x_hat = torch.hstack(x_hat)  # (B, T, C)
        return x_hat
