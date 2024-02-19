import torch.nn as nn


class Permute(nn.Module):
    def __init__(self, *dims):
        """`nn.Module` wrapper of `Tensor.permute`."""
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

    def __repr__(self):
        return f"Permute({self.dims})"


class View(nn.Module):
    def __init__(self, *shape, n_batch_dims: int = 1):
        """`nn.Module` wrapper of `Tensor.view`"""
        super().__init__()
        self.shape = shape
        self.n_batch_dims = n_batch_dims

    def forward(self, x):
        return x.view(*x.shape[0:self.n_batch_dims], *self.shape)

    def extra_repr(self):
        return f"{self.shape}, n_batch_dims={self.n_batch_dims}"


class AddConstant(nn.Module):
    def __init__(self, constant):
        """Adds a `constant` to the input"""
        super().__init__()
        self.constant = constant

    def forward(self, tensor1):
        return tensor1 + self.constant

    def __repr__(self):
        return f"AddConstant({self.constant})"
