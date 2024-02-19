import math

from numbers import Real
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def reduce(tensor: torch.Tensor, dim: Union[Tuple[int], int], operation: Callable = torch.sum):
    """Reduce a tensor along a dimension. Skips calling `operation` if tensor size along dim is 1."""
    if tensor.size(dim) == 1:
        return tensor.squeeze(dim)
    return operation(tensor, dim)


def gaussian_ll(y, mu, sd, epsilon: float = 1e-6, reduce_dim: Optional[int] = -1):
    """Compute Gaussian log-likelihood.

    Computes the likelihood element-wise without reduction. Dimensions must match or be broadcastable.

    Clamps the standard deviation at `epsilon` for numerical stability. This does not affect the gradient.

    Args:
        y (torch.Tensor): Targets (*)
        mu (torch.Tensor): Mean of the Gaussian (*)
        sd (torch.Tensor): Standard deviation of the Gaussian (*)
        epsilon (float, optional): Minimum standard deviation for numerical stability. Defaults to 1e-6.
        reduce_dim (int, optional): Dimension in y to reduce over. Defaults to -1 corresponding to D.

    Returns:
        torch.Tensor: Log-probabilities (*)
    """
    if epsilon:
        with torch.no_grad():
            sd = sd.clamp(min=epsilon)
    log_scale = math.log(sd) if isinstance(sd, Real) else sd.log()
    log_prob = -((y - mu) ** 2) / (2 * sd ** 2) - log_scale - 0.5 * math.log(2 * math.pi)
    return reduce(log_prob, reduce_dim) if reduce_dim else log_prob


def gaussian_mixture_ll(y, logits, mu, sd, epsilon: float = 1e-6, reduce_dim: int = -1):
    """Compute Gaussian Mixture log-likelihood.

    Clamps the standard deviation at `epsilon` for numerical stability. This does not affect the gradient.

    Args:
        y (torch.Tensor): Target (*, D)
        logits (torch.Tensor): Mixture probabilities/weights in log space (*, num_mix)
        mu (torch.Tensor): Mixture means (*, D, num_mix)
        sd (torch.Tensor): Mixture standard deviations (*, D, num_mix)
        epsilon (float, optional): Minimum standard deviation for numerical stability. Defaults to 1e-6.
        reduce_dim (int, optional): Dimension in y to reduce over. Defaults to -1 corresponding to D.

    Returns:
        torch.Tensor: Log-probabilities (*, D)
    """
    log_prob_y = gaussian_ll(y.unsqueeze(-1), mu, sd, epsilon=epsilon, reduce_dim=reduce_dim - 1)  # (*, D, num_mix)
    log_prob_mix = logits.log_softmax(-1)  # (*, num_mix)
    return torch.logsumexp(log_prob_y + log_prob_mix, dim=-1)  # (*, D)


def categorical_ll(y: torch.LongTensor, logits: torch.Tensor, reduce_dim: Optional[int] = -1):
    """Compute Categorical log-likelihood.

    A bit more memory efficient than instantiating a distributions.Categorical and using its log_prob method.

    Args:
        y (torch.LongTensor): Target values in [1, C-1] of any shape.
        logits (torch.Tensor): Event log-probabilities (unnormalized) of same shape (y.shape, C)

    Returns:
        torch.Tensor: Log-probabilities
    """
    logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    y = y.long().unsqueeze(-1)
    y, logits = torch.broadcast_tensors(y, logits)
    y = y[..., :1]
    log_prob = logits.gather(-1, y).squeeze(-1)
    return reduce(log_prob, reduce_dim) if reduce_dim else log_prob


def bernoulli_ll(y: torch.Tensor, logits: torch.Tensor, reduce_dim: Optional[int] = -1):
    """Compute Bernoulli log-likelihood

    Args:
        y (torch.Tensor): Target values in {0, 1} of any shape.
        probs (torch.Tensor): Event log-probabilities (unnormalized) of same shape as `y`.

    Returns:
        torch.Tensor: Log-probabilities
    """
    y, logits = torch.broadcast_tensors(y, logits)
    log_prob = -F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return reduce(log_prob, reduce_dim) if reduce_dim else log_prob


def discretized_logistic_ll(y: torch.Tensor, loc: torch.Tensor, log_scale: torch.Tensor, num_bins: int = 256, reduce_dim: Optional[int] = -1):
    """Log of the probability mass of the values y under the logistic distribution with parameters loc and scale.

    All dimensions are treated as independent.

    Assumes input data to be in `num_bins` equally-sized bins between -1 and 1.
    E.g. if num_bins=256, the 257 bin edges are: 
        -1, -254/256, ..., 254/256, 1  or 
        -1, -127/128, ..., 127/128, 1
    i.e. bin widths of 2/256 = 1/128.
    The data should not be exactly at the right bin edge (1).

    Noting that the CDF of the standard logistic distribution is simply the sigmoid function, we simply compute the
    probability mass under the logistic distribution per input element by using

        PDF(x_i | µ_i, s_i ) = CDF(x_i + 1/256 | µ_i, s_i) − CDF(x_i | µ_i, s_i ),

    where the locations µ_i and the log-scales log(s_i) are learned scalar parameters per input element and

        CDF(y | µ, s) = 1 / (1 + exp(-(y-µ)/s)) = Sigmoid((y-µ)/s)).
    
    We also use that

        log CDF(y | µ, s) = - Softplus((y - µ)/s)
        Softplus(y) = y - Softplus(-y)

    Args:
        y (torch.Tensor): targets to evaluate with shape (*).
        loc (torch.Tensor): loc of logistic distribution, shape (*) same as y.
        log_scale (torch.Tensor): log scale of distribution, shape (*) same as y, or either scalar or broadcastable.
        num_bins (int): number of bins, equivalent to specifying number of bits = log2(num_bins). Defaults to 256.
        reduce_dim (int, optional): Dimension in y to reduce over. Defaults to -1 corresponding to D.

    References:
        https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
        https://github.com/NVlabs/NVAE/blob/38eb9977aa6859c6ee037af370071f104c592695/distributions.py#L98
        https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py
    """
    # check input
    assert torch.max(y) <= 1.0 and torch.min(y) >= -1.0

    # compute y-µ and 1/s
    centered_y = y - loc
    inv_stdv = torch.exp(-log_scale)

    # compute CDF at left and right "bin edge" (floating) to compute total mass in between (cdf_delta)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_bins - 1))  # add half a bin width
    cdf_plus = torch.sigmoid(plus_in)
    minus_in = inv_stdv * (centered_y - 1.0 / (num_bins - 1))  # subtract half a bin width
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = cdf_plus - cdf_minus

    # log probability for edge case of 0 (mass from 0 to 0.5)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # equals log CDF(y+0.5) via softplus(y) = y - softplus(-y)

    # log probability for edge case of 255 (mass from 254.5 to 255)
    log_one_minus_cdf_minus = -F.softplus(minus_in)  # equals log 1 - CDF(y-0.5)

    # log probability in the center of the bin, to be used in extreme cases where cdf_delta is extremely small
    mid_in = inv_stdv * centered_y
    log_prob_mid = mid_in - log_scale - 2.0 * F.softplus(mid_in)  # = log PDF(y)
    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_prob_mid - math.log(num_bins / 2)
    )

    # handle edge cases
    log_prob = torch.where(y < 2 / num_bins - 1, log_cdf_plus, log_prob_mid_safe)  # edge case 0, y < -254/256
    log_prob = torch.where(y > 1 - 2 / num_bins, log_one_minus_cdf_minus, log_prob)  # edge case 255, y > 254/256
    return reduce(log_prob, reduce_dim) if reduce_dim else log_prob


# @torch.jit.script
def discretized_logistic_mixture_ll(
    y: torch.Tensor,
    logit_probs: torch.Tensor,
    locs: torch.Tensor,
    log_scales: torch.Tensor,
    num_bins: int = 256,
    reduce_dim: int = -1,
):
    """Compute log-likelihood for a mixture of discretized logistics.

    The implementation is partially as in https://arxiv.org/abs/1701.05517 but does not assume
    three RGB colour channels nor does it condition them on each other (as described in Section 2.2).
    Hence, the channels, and all other dimensions, are regarded as independent.

    For more details, refer to documentation for `discretized_logistic_ll`.

    Args:
        y (torch.Tensor): Targets (*, D).
        logit_probs (torch.Tensor): Unnormalized log probabilities of mixture components (*, num_mix).
        locs (torch.Tensor): Location parameters of mixture components (*, D, num_mix).
        log_scales (torch.Tensor): Scale parameters of mixture components in log-space (*, D, num_mix).
        num_bins (int): Number of bins for the quantization.
        reduce_dim (int, optional): Dimension in y to reduce over. Defaults to -1 corresponding to D.
    """
    # check input
    assert torch.max(y) <= 1.0 and torch.min(y) >= -1.0

    # repeat y for broadcasting to mixture dim
    num_mix = logit_probs.size(-1)
    y = y.unsqueeze(-1).expand(*[-1] * y.ndim, num_mix)  # (*, D, num_mix)

    # compute y-µ and 1/s
    centered_y = y - locs
    inv_stdv = torch.exp(-log_scales)

    # compute CDF at left and right "bin edge" (floating) to compute total mass in between (cdf_delta)
    plus_in = inv_stdv * (centered_y + 1.0 / (num_bins - 1))
    cdf_plus = torch.sigmoid(plus_in)
    minus_in = inv_stdv * (centered_y - 1.0 / (num_bins - 1))
    cdf_minus = torch.sigmoid(minus_in)
    cdf_delta = cdf_plus - cdf_minus

    # log probability for edge case of 0 (mass from 0 to 0.5)
    log_cdf_plus = plus_in - F.softplus(plus_in)  # = log CDF(y+0.5) via softplus(y) = y - softplus(-y)

    # log probability for edge case of 255 (mass from 254.5 to 255)
    log_one_minus_cdf_minus = -F.softplus(minus_in)  # = log 1 - CDF(y-0.5)

    # log probability in the center of the bin, to be used in extreme cases where cdf_delta is extremely small
    mid_in = inv_stdv * centered_y
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)
    log_prob_mid_safe = torch.where(
        cdf_delta > 1e-5, torch.log(torch.clamp(cdf_delta, min=1e-10)), log_pdf_mid - math.log(num_bins / 2)
    )

    # handle edge cases
    log_prob = torch.where(y < 2 / num_bins - 1, log_cdf_plus, log_prob_mid_safe)  # edge case 0, y < -254/256
    log_prob = torch.where(y > 1 - 2 / num_bins, log_one_minus_cdf_minus, log_prob)  # edge case 255, y > 254/256

    log_prob = reduce(log_prob, reduce_dim - 1)  # Reduce data dimension per component
    log_prob = log_prob + torch.log_softmax(logit_probs, dim=-1)  # 
    return torch.logsumexp(log_prob, dim=-1)  # Normalize over mixture components (in log-prob space)
