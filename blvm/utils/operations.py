import math

from typing import Any, Tuple, Union

import torch

from torch import Tensor
from torchtyping import TensorType

from blvm.utils.padding import padding2argument



def stack_tensor(x: TensorType[..., "D"], stack_size: int, dim: int = -1) -> Tuple[TensorType[..., "D/S", "S"]]:
    """Split a tensor into stacks of `stack_size` along `dim` and return it with a new right-most stack dimension"""
    if abs(dim) > x.ndim:
        raise ValueError(f"Got {dim=} which is out of range for x with shape {x.shape}")

    dim = dim if dim > 0 else x.ndim + dim

    padding = stack_size - x.size(dim) % stack_size
    if padding == stack_size:
        padding = 0

    if padding:
        pad_arg = padding2argument(x.ndim, dim - 1, pad_right=padding)
        x = torch.nn.functional.pad(x, pad_arg)

    new_shape = [x.size(i) if i != dim else int(x.size(i) / stack_size) for i in range(x.ndim)] + [stack_size]
    x = x.view(*new_shape)

    return x, padding


def unstack_tensor(
    x: TensorType[..., "D/S", "S"], stack_size: int, padding: int = 0, dim: int = -1
) -> TensorType[..., "D"]:
    """Reverse the `stack_tensor` operation by collapsing the stack dimension at `dim` into the preceding dimension"""
    if abs(dim) > x.ndim:
        raise ValueError(f"Got {dim=} which is out of range for x with shape {x.shape}")

    dim = dim if dim > 0 else x.ndim + dim

    new_shape = [x.size(i) if i != (dim - 1) else int(x.size(i) * stack_size) for i in range(x.ndim)]
    new_shape[-1] = -1

    x = x.view(*new_shape)

    if padding:
        padding = padding2argument(x.ndim, dim - 1, pad_right=-padding)
        x = torch.nn.functional.pad(x, padding)

    return x


def reverse_sequences(x: torch.Tensor, x_sl: torch.Tensor, batch_first: bool = False):
    """Reverse a sequence keeping right padding untouched and in position (on the right).

    Note: This method only works with right padding (not left padding or a combination).
    Note: An alternative option is to first use pack_padded_sequence and then reverse that.

    Args:
        x (torch.Tensor): Padded sequences to reverse (T, B, *) (or (B, T, *) if `batch_first == True`)
        x_sl (torch.Tensor): Sequence lengths

    Returns:
        torch.Tensor: Sequences reversed along time axis but with same padding as before
    """
    if batch_first:
        x = x.permute(1, 0)

    max_len = x_sl.max()
    padding = (max_len - x_sl).unsqueeze(0).to(x.device)
    forward_ids = torch.arange(0, max_len, 1, device=x.device).expand(x.size(1), -1).permute(1, 0)
    reverse_ids = torch.arange(max_len - 1, -1, -1, device=x.device).expand(x.size(1), -1).permute(1, 0) - padding

    # Do not reverse padding
    mask = reverse_ids < 0
    reverse_ids[mask] = forward_ids[mask]

    # Match shape with x as a view
    x_shape_singular_dims = reverse_ids.shape[:2] + (1,) * (x.ndim - 2)  # (T, B, 1, 1, ...)
    reverse_ids = reverse_ids.view(x_shape_singular_dims).expand(-1, -1, *x.shape[2:])  # (T, B, *x.shape[2:])
    out = torch.gather(x, 0, reverse_ids)
    if batch_first:
        return out.permute(1, 0)
    return out


def sequence_mask(
    seq_lens: Union[list, torch.Tensor],
    stride: int = 1,
    max_len: int = None,
    dtype: torch.dtype = torch.bool,
    device: torch.device = None,
):
    """
    Creates a binary sequence mask where all entries up to seq_lens are 1 and the remaining are 0.

    Args:
        seq_lens (Tensor): The sequence lengths from which to construct the mask. Should be shape N with dtype == int64.
        stride (int):
        max_len (int): The temporal dimension of the sequence mask. If None, will use max of seq_lens.
        dtype (torch.dtype): The type of the mask. Default is torch.bool.

    Returns:
        Tensor: The sequence mask of shape (N, T).
    """
    if isinstance(seq_lens, torch.Tensor):
        device = seq_lens.device if device is None else device
        if device != seq_lens.device:
            seq_lens = seq_lens.to(device)
    else:
        seq_lens = torch.tensor(seq_lens, device=device, dtype=int)

    N = seq_lens.size(0)
    T = max_len or math.ceil(seq_lens.max() / stride)
    seq_mask = torch.arange(T, device=device).unsqueeze(0).repeat((N, 1)) < seq_lens.unsqueeze(1)
    return seq_mask.to(dtype)


def split_sequence(
    x: torch.Tensor,
    x_sl: torch.Tensor,
    length: int,
    overlap: int = 0,
    drop_inactive: bool = True,
    mode: str = "consume",
):
    """Split a sequence into a number of subsequences of given length, optionally with overlap.

    `mode == "consume"`:
         subsequence length = `length`
        subsequence overlap = `overlap`
         new values in each = `length - overlap`

    `mode == "extend"`:
         subsequence length = `length + overlap`
        subsequence overlap = `overlap`
         new values in each = `length`

    In the `consume` mode, the `overlap` consumes from the number of new values in the subsequence of `length`.
    In the `extend` mode, the `overlap` extends the `length` to always have `length` new values in each subsequence.

    The `consume` mode is only supported for `overlap < length`. If `overlap >= length`, we cannot create subsequences
    that simultaneously satisfy having length given by `length` and overlap with the previous subsequence with `overlap`
    values. This would result in `length - overlap <= 0` new values in each subsequence compared to the former.

    In the case where `overlap < length`, and in general, the `extend` mode is supported which ensures `length` new
    values in each subsequence by extending the length of each subsequence with the amount of `overlap` leading to a
    total length of `length + overlap`

    Args:
        x (torch.Tensor): Tensor of shape (B, T, *) to split along T.
        x_sl (torch.Tensor): Tensor of shape (B,) with the sequence lengths of each example in x.
        length (int): Length of the subsequences to create. Last subsequence will contain the remainder.
        overlap (int): Amount of overlap to have between two neighbouring subsequences.
        drop_inactive (bool): If True, remove short examples from the batch as we exceed their sequence length.
        mode (str): Mode of the splitting with `consume` and `extend` options. Defaults to `consume`.
    """
    if mode == "consume":
        if overlap >= length:
            raise ValueError("`split_sequence` does not support `overlap >= length` in `consume` mode")
        max_num_splits = math.ceil(x.size(1) / (length - overlap))  # maximum possible number of splits
        start_idx = [i * (length - overlap) for i in range(max_num_splits)]
        stop_idx = [s + length for s in start_idx]
    elif mode == "extend":
        max_num_splits = math.ceil(x.size(1) / length)
        start_idx = [max(i * length - overlap, 0) for i in range(max_num_splits)]
        stop_idx = [(i + 1) * length for i in range(max_num_splits)]
    else:
        raise ValueError("Unknown mode `{mode}`. Recognized options are `consume` and `extend`.")

    active_examples_idx = torch.ones(x.shape[0]).to(bool)
    splits_x = []
    splits_x_sl = []
    i = 0
    while active_examples_idx.any():
        if drop_inactive:
            split_x = x[active_examples_idx, start_idx[i] : stop_idx[i]]
        else:
            split_x = x[:, start_idx[i] : stop_idx[i]]

        new_active_examples_idx = x_sl > stop_idx[i]  # stop idx is exclusive and x_sl is correspondingly not an index.

        split_x_sl = length * new_active_examples_idx + (x_sl - start_idx[i]).clamp(0) * ~new_active_examples_idx
        if drop_inactive:
            split_x_sl = split_x_sl[active_examples_idx]

        active_examples_idx = new_active_examples_idx

        splits_x.append(split_x)
        splits_x_sl.append(split_x_sl)

        i += 1

    return splits_x, splits_x_sl


def update_running_variance(
    mean_a: Union[torch.Tensor, float],
    weight_a: Union[torch.Tensor, float],
    M2_a: Union[torch.Tensor, float],
    mean_b: Union[torch.Tensor, float] = 0,
    weight_b: Union[torch.Tensor, float] = 0,
    M2_b: Union[torch.Tensor, float] = 0,
):
    r"""Online variance update c.f. parallel variance algorithm at [1].

    The required inputs are the current average, its weight and the sum of squared differences to the previous average.
    These can be computed as follows:

    math:`\bar{a}_i = \frac{1}{{n_i}} \sum_j^{n_i} a_{i,j}`
    math:`M_{2,i} = \sum_j^{n_i} (a_{i,j} - \bar{a}_{i-1}) ** 2`

    where math:`j` loops over dimension(s) of interest in the data math:`a` and math:`i` is the iteration index.

    In Torch:
    ```python
    mean_a = torch.mean(a, dim=dim)
    weight_a = a.size(dim)
    M2_a = torch.sum((a - mean_a) ** 2, dim=dim)
    ```

    Initial values for `b` if starting an iteration with just `a` observed:

    math:`\bar{b}_{i=0} = 0`
    math:`n_i = 0`
    math:`M_{2,i=0} = 0`

    NOTE This can be extended to also compute higher order statistics (skewness and kurtosis) c.f. [2]
    NOTE Can be simplified in the "incremental case" where there is a constant weight/number of observations in `b`.

    [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    [2] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Higher-order_statistics
    """
    weight = weight_a + weight_b
    delta = mean_b - mean_a
    M2 = M2_a + M2_b + delta ** 2 * (weight_a * weight_b / weight)
    var = M2 / (weight - 1)
    mean = (weight_a * mean_a + weight_b * mean_b) / weight
    return var, mean, weight, M2


def detach(x: Union[torch.Tensor, Any]):
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()

    return x


def detach_to_device(x: Any, device: torch.device):
    """Detach a tensor from the computational graph, clone and place it on the given device"""
    if x is None:
        return None

    if isinstance(x, torch.Tensor) or isinstance(x, torch.nn.Module):
        return x.detach().clone().to(device)

    return torch.tensor(x, device=device, dtype=torch.float)
