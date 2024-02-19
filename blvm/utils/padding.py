"""Methods relating to padding"""

import numpy as np
import torch


def padding2argument(n_dims: int, dim: int, pad_left: int = 0, pad_right: int = 0, as_array: bool = False):
    """Return left/right padding along single `dim` in the format used by `torch.nn.functional.pad`"""
    dim = dim if dim > 0 else n_dims + dim
    padding = [0, 0] * (n_dims - dim - 1) + [pad_left, pad_right]
    if as_array:
        np.asarray(padding)
    return tuple(padding)


def split_padding_sym_asym(padding: int):
    """Return the symmetric and asymmetric parts of the padding"""
    sym = padding // 2
    asym = padding - 2 * sym
    return sym, asym


def split_padding_left_right(padding: int, side: str = "both"):
    """Split an amount of `padding` into left and right parts according to `side` in ['left', 'right', 'both']
    returning a tuple of the resulting left and right padding"""
    if side == "left":
        return padding, 0
    elif side == "right":
        return 0, padding
    elif "both":
        sym, asym = split_padding_sym_asym(padding)
        return sym // 2, sym // 2 + asym
    raise ValueError(f"Unknown side `{side=}`. Valid options are `left`, `right` and `both`")


def pad_to_length(x: torch.Tensor, length: int, pad_side: str = "right", dim: int = -1):
    """Return the tensor padded to at least `length` long along dimension `dim` with padding on `pad_side`"""
    dim = dim if dim > 0 else x.ndim + dim  # get absolute dimension
    p = get_length_padding(x.size(dim), length)
    if not p:
        return x
    
    pad_left, pad_right = split_padding_left_right(p, pad_side)
    padding = padding2argument(x.ndim, dim, pad_left, pad_right)
    return torch.nn.functional.pad(x, padding)


def pad_modulo(x: torch.Tensor, stride: int, kernel_size: int = 0, dilation: int = 1, pad_to_kernel_size: bool = False, pad_side: str = "left", dim: int = -1):
    """Pad a tensor along `dim` such that `(i + p - k) mod s == 0` where `i=x.size(dim)`"""
    dim = dim if dim > 0 else x.ndim + dim  # get absolute dimension

    p = get_modulo_padding(x.shape[dim], stride, kernel_size, dilation, as_array=False, pad_to_kernel_size=pad_to_kernel_size)
    pad_left, pad_right = split_padding_left_right(p, pad_side)
    padding = padding2argument(x.ndim, dim, pad_left, pad_right)
    return torch.nn.functional.pad(x, padding)


def pad_same(x: torch.Tensor, stride: int, kernel_size: int = 0, dilation: int = 1, pad_side: str = "left", dim: int = -1):
    """Pad a tensor along `dim` such that its convolution has size `ceil(x.size(dim) / stride)` along `dim`."""
    dim = dim if dim > 0 else x.ndim + dim  # get absolute dimension
    p = get_same_padding(x.shape[dim], stride, kernel_size, dilation)
    pad_left, pad_right = split_padding_left_right(p, pad_side)
    padding = padding2argument(x.ndim, dim, pad_left, pad_right)
    return torch.nn.functional.pad(x, padding)


def get_length_padding(actual_length: int, minimum_length: int):
    return max(minimum_length - actual_length, 0)


def get_modulo_padding(length: int, stride: int, kernel_size: int = 0, dilation: int = 1, as_array: bool = False, pad_to_kernel_size: bool = False):
    """Return `padding` such that `length + padding` is evenly divisible by `stride`, optionally given a `kernel_size`.

    Mathematically, this method returns `p` such that `(i + p - k) mod s == 0` (where `i = length`).

    The output length after convolution will be minimum 1.
    """
    if dilation > 1:
        raise NotImplementedError(f"Dilation greater than 1 not yet supported but got {dilation=}.")

    if length < kernel_size:
        if pad_to_kernel_size:
            return kernel_size - length
        raise ValueError(f"Input {length=} was shorter than {kernel_size=} and {pad_to_kernel_size=}.")

    missing = (length - kernel_size) % stride
    padding = stride - missing if missing else 0

    if as_array:
        return np.asarray(padding)
    return padding


def get_modulo_length(length: int, stride: int, kernel_size: int = 0):
    """Return the smallest number larger than `length` that it is evenly divisible by `stride` 
    optionally given a `kernel_size`"""
    return length + get_modulo_padding(length, stride, kernel_size)


def get_same_padding(length: int, stride: int, kernel_size: int, dilation: int = 1, as_array: bool = True):
    """Return the padding to add to `length` to yield a length of the convolved input of `ceil(length / stride).`
    
    This also ensures that after padding the input is wholly strideable `(i + p - k) mod s == 0` (where `i = length`)
    except in the rare case highlighted below.

    Note:
        The padded input is not wholly strideable in the case where `k < s` and the total returned `padding` is zero.
        Total padding is zero when `dilation * (kernel_size - 1) - (length - 1) % stride <= 0`.
        This is likely the case when `k < s` and becomes more likely the larger the difference between `k` and `s`.
        This is not likely to be a problem for two reasons:
            1. wwith `k < s` many inputs are already being ignored in the convolution so ignoring a few more :shrug:
            2. using `k < s` is generally rare and hence this won't be triggered that often.
    
    TODO Add `transposed` argument and document the corresponding guarantee on the length of the convolved input.
    """
    # Pad by what is consumed by a (dilated) convolution `d(k-1)` minus what is "added" (not convolved) by the stride.
    return max(0, dilation * (kernel_size - 1) - (length - 1) % stride)


def get_same_padding_transposed(kernel_size: int, stride: int, dilation: int = 1, as_array: bool = True):
    return dilation * (kernel_size - 1) + 1 - stride


def get_same_length(length: int, kernel_size: int, stride: int, dilation: int = 1):
    """"""
    sym_pad, unsym_pad = get_same_padding(length, kernel_size, stride, dilation, as_array=False)
    padding = 2 * sym_pad + unsym_pad
    return length // 2 + padding
