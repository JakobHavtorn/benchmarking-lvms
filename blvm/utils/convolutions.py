from collections.abc import Iterable
from itertools import repeat
from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn as nn


def _ntuple(n):
    """Given an integer, return that integer in an n-tuple. Given an Iterable, return that directly instead"""

    def parse(x):
        """Given an integer, return that integer in an n-tuple. Given an Iterable, return that directly instead"""
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)


def get_same_padding(
    convolution: nn.Module = None,
    kernel_size: Union[int, Tuple[int]] = None,
    stride: Union[int, Tuple[int]] = None,
    dilation: Union[int, Tuple[int]] = None,
    transposed: Optional[bool] = False,
    in_shape: tuple = None,
):
    """
    Return the padding to apply to a given convolution such as it reproduces the 'same' behavior from Tensorflow

    This also works for pooling layers.

    For transposed convolutions, the symmetric padding is always returned as None.

    Args:
        in_shape (tuple): Input tensor shape excluding batch (D1, D2, ...)
        convolution (nn.Module): Convolution module object
    returns:
        sym_padding, unsym_padding: Symmetric and unsymmetric padding to apply. We split in two because nn.Conv only
                                    allows setting symmetric padding so unsymmetric has to be done manually.
    """
    if convolution is not None:
        # assert convolution.transposed or in_shape is not None, "in_shape is required for non-tranposed convolutions"
        kernel_size = np.asarray(convolution.kernel_size)
        dilation = np.asarray(convolution.dilation) if hasattr(convolution, "dilation") else 1
        stride = np.asarray(convolution.stride)
        transposed = convolution.transposed
    else:
        kernel_size = np.asarray(_single(kernel_size))
        dilation = np.asarray(_single(dilation)) if dilation is not None else 1
        stride = np.asarray(_single(stride)) if stride is not None else 1

    if not transposed:
        if in_shape is not None:
            assert len(in_shape) == len(kernel_size), "`in_shape` tensor is not the same dimension as the kernel"
            # in_shape = np.asarray(in_shape)
            output_size = (in_shape - 1) // stride + 1
            padding_input = np.maximum(0, (output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - in_shape)
            # padding_input = np.maximum(0, in_shape - 1 + (kernel_size - 1) * dilation + 1 - in_shape)
            # padding_input = np.maximum(0, (kernel_size - 1) * dilation)
            # padding_input = (kernel_size - 1) * dilation
            odd_padding = padding_input % 2 != 0
            sym_padding = tuple(padding_input // 2)
            unsym_padding = [y for x in odd_padding for y in [0, int(x)]]
        else:
            assert kernel_size % 2 == 1, "Non-transposed convolutions require `in_shape` to be given if kernel is even."
            sym_padding = tuple(kernel_size // 2)
            unsym_padding = None
    else:
        padding_input = kernel_size - stride
        sym_padding = None
        unsym_padding = [
            y for x in padding_input for y in [-int(np.floor(int(x) / 2)), -int(np.floor(int(x) / 2) + int(x) % 2)]
        ]

    return sym_padding, unsym_padding


def compute_conv_attributes_single(i=0, k=np.nan, p=np.nan, s=np.nan, d=1, s_in=1, r_in=1, start_in=0):
    """Computes the output channels and receptive field of a (potentially N-dimensional) convolution.

    To calculate the receptive field in each layer, besides the number of features n in each dimension, we need to
    keep track of some extra information for each layer. These include the current receptive field size r, the
    distance between two adjacent features ("effective stride") s, and the center coordinate of the upper left
    feature (the first feature) start.

    Note that the center coordinate of a feature is defined to be the center coordinate of its receptive field.

    All the arguments default to nan since, in order to get any of them at the output layer, for instance the receptive
    field, not all are required and nan allows running all calculations regardless.

    The requirements are:
        o_out:      i, k, p, s
        s_out:      s_in, s
        r_out:      r_in, s_in, k, s
        start_out:  start_in, i, s_in, k, p, s

    Args:
        i (int or np.ndarray of int): Number of features
        k (int or np.ndarray of int): Kernel size
        p (int or np.ndarray of int): Amount of padding applied
        s (int or np.ndarray of int): Size of the stride used
        s_in (int): Distance between the centers of two adjacent features
        r_in (int): Receptive field of a feature.
        start_in (float): Position of the first feature's receptive field in layer i. Defaults to 0.
                          (index starts from 0, negative means center is in the padding)

    Returns:
        tuple: (o_out, s_out, r_out, start_out) corresponding to the above but mapped through this conv layer.

    [1] https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
    """
    s_out = s_in * s
    k_eff = k + (k - 1) * (d - 1)  # k if d==1
    r_out = r_in + (k_eff - 1) * s_in
    o_out = ((i - k_eff + 2 * p) // s) + 1
    actual_padding = (o_out - 1) * s - i + k_eff
    pad_left = actual_padding // 2
    start_out = start_in + ((k_eff - 1) / 2 - pad_left) * s_in
    return o_out, s_out, r_out, start_out


def compute_conv_attributes(
    kernels: List[Union[tuple, int]],
    paddings: List[Union[tuple, int]],
    strides: List[Union[tuple, int]],
    dilations: List[Union[tuple, int]] = None,
    in_shape: Union[tuple, int] = 0,
    stride_in: Union[tuple, int] = 1,
    receptive_field_in: Union[tuple, int] = 1,
    start_in: Union[tuple, int] = 0,
    return_all: bool = False
):
    """Repeatedly applies `compute_conv_attributes_single` to multiple consecutive N-dimensional convolutions.
    
    Example
        kernels = [2, 2, 2]
        paddings = [0, 0, 0]
        strides = [1, 1, 1]
        dilations = [1, 1, 2]
        o, s, r, c = compute_conv_attributes(kernels, paddings, strides, dilations)

    Args:
        in_shape (tuple or int): Input length
        kernels (list of tuple or int): Kernel sizes
        paddings (list of tuple or int): Padding sizes
        strides (list of tuple or int): Stride sizes
        stride_in (tuple or int, optional): Distance between the centers of two adjacent features. Defaults to 1.
        receptive_field_in (tuple or int, optional): Receptive field of a feature. Defaults to 1.
        start_in (tuple or int, optional): Position of the first feature's receptive field in layer i. Defaults to 0.
                                           (index starts from zero, negative means center is in the padding).
        return_all (bool): If True, returns the attributes at every convolution not just the last. Defaults to False.

    Returns:
        tuple: (out_shape, jump_out, receptive_field_out, start_out) corresponding to the above but mapped through this conv layer.
    """
    # default dilations
    if dilations is None:
        dilations = [1] * len(kernels)

    # Check inputs
    assert len(kernels) == len(paddings) == len(strides), "Number of layers in each of the parameters must be equal"

    all_n_dims = {len(kernels[0])} if isinstance(kernels[0], tuple) else {1}
    for k, p, s, d in zip(kernels, paddings, strides, dilations):
        all_n_dims.add(len(k) if isinstance(k, tuple) else 1)
        all_n_dims.add(len(p) if isinstance(p, tuple) else 1)
        all_n_dims.add(len(s) if isinstance(s, tuple) else 1)
        all_n_dims.add(len(d) if isinstance(d, tuple) else 1)

    if len(all_n_dims) != 1:
        msg = f"Must give only tuples (or ints) of same dimensions but got different dimensions: {all_n_dims}"
        raise ValueError(msg)

    # Chosen dimensionality is the maximum of the two encountered (i.e. 1 or N)
    n_dims = max(all_n_dims)

    # Convert all parameters from a mix of ints and tuples to tuples (arrays) of the same dimensionality
    in_shape = np.array(_ntuple(n_dims)(in_shape))

    stride_in = np.array(_ntuple(n_dims)(stride_in))
    receptive_field_in = np.array(_ntuple(n_dims)(receptive_field_in))
    start_in = np.array(_ntuple(n_dims)(start_in))

    kernels = [np.array(_ntuple(n_dims)(k)) for k in kernels]
    paddings = [np.array(_ntuple(n_dims)(k)) for k in paddings]
    strides = [np.array(_ntuple(n_dims)(k)) for k in strides]
    dilations = [np.array(_ntuple(n_dims)(k)) for k in dilations]

    out_shape = in_shape
    all_attributes = []
    for k, p, s, d in zip(kernels, paddings, strides, dilations):
        out_shape, stride_in, receptive_field_in, start_in = compute_conv_attributes_single(
            out_shape, k, p, s, d, stride_in, receptive_field_in, start_in
        )

        if n_dims > 1:    
            attrs = (tuple(out_shape.tolist()), tuple(stride_in.tolist()), tuple(receptive_field_in.tolist()), tuple(start_in.tolist()))
        else:
            attrs = (out_shape[0], stride_in[0], receptive_field_in[0], start_in[0])
        all_attributes.append(attrs)

    if return_all:
        o, s, r, c = list(map(list, zip(*all_attributes)))  # transpose list of lists
        return o, s, r, c
    return all_attributes[-1]
