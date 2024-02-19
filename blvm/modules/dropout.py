from typing import Union

import torch
import torch.nn as nn
import torch.jit as jit

from torchtyping import TensorType


class Dropout1dPackedData(nn.Module):
    def __init__(self, p):
        """
        Uses a fixed dropout mask across time steps.

        Args:
            p (float): Probability of an element being zeroed
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        """
        Args:
            x (Tensor): Should be 2D of shape TF.

        Returns:
            Tensor: The x with zeroed out features along the temporal dimension.
        """
        if not self.training:
            return x

        assert x.ndim == 2, 'Expected a 2D tensor of shape TF (i.e., data from PackedSequence).'

        F = x.size(1)
        do_mask = (torch.rand([1, F], device=x.device) > self.p).to(torch.float) * (1.0 / (1.0 - self.p))
        x = x * do_mask
        return x

    def extra_repr(self):
        return f"Dropout1dPacked(p={self.p})"


class WordDropout(jit.ScriptModule):
    def __init__(self, dropout_rate: float = 0.0, mask_value: Union[float, int] = 0, mask_first_timestep: bool = False):
        """Dropout module that masks out all features for any sampled timestep.

        Args:
            dropout_rate (float, optional): Probability of sampling a timestep for dropout. Defaults to 0.0.
            mask_value (Union[float, int], optional): Value to use for dropout masking. Defaults to 0.
            mask_first_timestep (bool, optional): If `False`, do not mask out the first timestep. Defaults to False.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.mask_value = mask_value
        self.mask_first_timestep = mask_first_timestep

    def forward(self, x: TensorType["B", "T", -1]):
        """Dropout entire timesteps in x of shape (B, T, *D)"""
        if self.training and self.dropout_rate > 0:
            mask = torch.bernoulli(torch.full((x.size(0), x.size(1)), self.dropout_rate, device=x.device)).to(bool)
            mask[:, 0] = self.mask_first_timestep
            x = x.clone()  # We can't modify x in-place
            x[mask, ...] = self.mask_value

        return x

    def __repr__(self):
        return f"WordDropout(dropout_rate={self.dropout_rate}, mask_value={self.mask_value}, mask_first_timestep={self.mask_first_timestep})"
