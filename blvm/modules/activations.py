import torch
import torch.nn as nn


class GatedTanhUnit(nn.Module):
    """Gated Tanh activation as in PixelCNN and WaveNet"""
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, self.dim)
        return torch.tanh(a) * torch.sigmoid(b)
