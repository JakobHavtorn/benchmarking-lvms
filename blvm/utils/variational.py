import math

from typing import Optional, Union, Tuple

import torch
import torch.jit as jit

from torchtyping import TensorType

from blvm.utils.log_likelihoods import gaussian_ll


def kl_divergence(q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution):
    """Compute Kullback-Leibler divergence KL(q||p) between two distributions.

        KL(q||p) = \int q(x) \log [q(x) / p(x)] dx = - \int q(x) \log [p(x) / q(x)] dx

    Note that the order of the distributions q and p is flipped compared to the usual order.
    This is done since KL(q||p) is the order used in the ELBO.

    The usual order (which is NOT used here) is

        KL(p || q) = \int p(x) \log [p(x) / q(x)] dx = - \int p(x) \log [q(x) / p(x)] dx

    Consider two probability distributions P and Q.
    Usually, P represents the data, the observations, or a measured probability distribution.
    Distribution Q represents instead a theory, a model, a description or an approximation of P.

    The Kullback–Leibler divergence is then interpreted as the average extra number of bits
    required to encode samples of P using a code optimized for Q rather than one optimized for P.

    Args:
        q_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.
        p_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape` in units of nats.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    return torch.distributions.kl_divergence(q_distrib, p_distrib)


def kl_divergence_mc(
    q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution, z: torch.Tensor
):
    """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

    Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

    Args:
        z: Sample or samples from the variational distribution `q_distrib`.
        q_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.
        p_distrib (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        tuple: KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
    """
    q_logprob = q_distrib.log_prob(z)
    p_logprob = p_distrib.log_prob(z)
    kl_dwise = q_logprob - p_logprob
    return kl_dwise, q_logprob, p_logprob


@jit.script
def kl_divergence_gaussian(mu_q: torch.Tensor, sd_q: torch.Tensor, mu_p: torch.Tensor, sd_p: torch.Tensor):
    """Elementwise analytical KL divergence between two Gaussian distributions KL(q||p) (no reduction applied)."""
    return sd_p.log() - sd_q.log() + (sd_q.pow(2) + (mu_q - mu_p).pow(2)) / (2 * sd_p.pow(2)) - 0.5


def kl_divergence_gaussian_mc(
    mu_q: torch.Tensor,
    sd_q: torch.Tensor,
    mu_p: torch.Tensor,
    sd_p: torch.Tensor,
    z: torch.Tensor,
    epsilon: float = 0,
    reduce_dim: Optional[int] = None,
):
    """Elementwise MC KL divergence between two Gaussian distributions KL(q||p) (no reduction applied)."""
    return gaussian_ll(z, mu_q, sd_q, epsilon, reduce_dim) - gaussian_ll(z, mu_p, sd_p, epsilon, reduce_dim)


def discount_free_nats(
    kld: TensorType["B":..., "shared":...],
    free_nats: float = None,
    shared_dims: Union[Tuple[int], int] = None,
) -> torch.Tensor:
    """Free bits as introduced in [1] but renamed to free nats because that's what it really is with log_e.

    In the paper they divide all latents Z into K groups. This implementation assumes a that each KL tensor passed
    to __call__ is one such group.

    By default, this method discounts `free_nats` units of nats elementwise in the KL regardless of its shape.

    If the KL tensor has more dimensions than the batch dimension, the free_nats budget can be optionally
    shared across those dimensions by setting `shared_dims`. E.g. if `kld.shape` is (32, 10) and `shared_dims` is -1,
    each of the 10 elements in the last dimension will get 1/10th of the free nats budget. If `kld.shape` is (32, 10, 10)
    and `shared_dims` is (-2, -1) each of the 10*10=100 elements will get 1 / 100th.

    The returned KL with `free_nats` discounted is equal to max(kld, freebits_per_dim)

    [1] https://arxiv.org/pdf/1606.04934
    """
    if free_nats is None or free_nats == 0:
        return kld

    if isinstance(shared_dims, int):
        shared_dims = (shared_dims,)

    # equally divide free nats budget over the elements in shared_dims
    if shared_dims is not None:
        n_elements = math.prod([kld.shape[d] for d in shared_dims])
        min_kl_per_dim = free_nats / n_elements
    else:
        min_kl_per_dim = free_nats

    min_kl_per_dim = torch.tensor(min_kl_per_dim, dtype=kld.dtype, device=kld.device)
    freenats_kl = torch.maximum(kld, min_kl_per_dim)
    return freenats_kl


@jit.script
def precision_weighted_gaussian(
    mu_1: torch.Tensor,
    sd_1: torch.Tensor,
    mu_2: torch.Tensor,
    sd_2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return an updated Gaussian by precision weighting two Gaussians, akin to a Bayesian posterior update"""
    pr_1 = sd_1.pow(-2)
    pr_2 = sd_2.pow(-2)
    va_w = (pr_1 + pr_2).pow(-1)
    mu_w = va_w * (mu_1 * pr_1 + mu_2 * pr_2)
    sd_w = va_w.sqrt()
    return mu_w, sd_w


@jit.script
def rsample_gaussian(mu: torch.Tensor, sd: torch.Tensor):
    """Return a reparameterized sample from a given Gaussian distribution.

    Args:
        mu (torch.Tensor): Gaussian mean of shape (*)
        sd (torch.Tensor): Gaussian standard deviation of shape (*) or (1)

    Returns:
        torch.Tensor: Reparameterized sample of shape (*)
    """
    return torch.randn_like(mu).mul(sd).add(mu)


# @jit.script
def rsample_gaussian_mixture(
    logits: torch.Tensor,
    mu: torch.Tensor,
    sd: torch.Tensor,
    eps: float = 1e-6,
    rsample_categorical: bool = False,
    tau: float = 1.0,
):
    """Return a reparameterized sample from a given Gaussian Mixture Model.

    Args:
        logits (torch.Tensor): (*, num_mix)
        mu (torch.Tensor): (*, D, num_mix)
        sd (torch.Tensor): (*, D, num_mix)
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.
        rsample_categorical (bool): If True, sample the mixture indicator differentiably using Gumbel-Softmax trick.
        tau (float): Temperature for Gumbel sampling. Has no effect if `rsample_categorical` is False.

    Returns:
        torch.Tensor: Sample from the GMM `(*, D)`

    NOTE Is the sampling of the categorical "wrong"? Should we rsample that as well?
    """
    # sample mixture indicator from categorical
    if rsample_categorical:
        argmax = rsample_gumbel_softmax(logits, hard=True, tau=tau, return_argmax=True)
    else:
        gumbel = -torch.log(-torch.log(torch.empty_like(logits).uniform_(eps, 1.0 - eps)))  # (*, num_mix)
        argmax = torch.argmax(logits + gumbel, dim=-1, keepdim=True)  # (*, 1)

    # broadcast argmax onto mu and sd shapes (including D)
    argmax = argmax.expand(*argmax.shape[:-1], mu.size(-2)).unsqueeze(-1)  # (*, D, 1)

    # select gaussian component
    mu = torch.gather(mu, index=argmax, dim=-1).squeeze(-1)
    sd = torch.gather(sd, index=argmax, dim=-1).squeeze(-1)

    # sample from gaussian
    x = rsample_gaussian(mu, sd)
    return x


@jit.script
def rsample_gumbel(mean: torch.Tensor, scale: torch.Tensor, fast: bool = True, eps: float = 1e-10):
    """Sample from a Gumbel distribution using the inverse CDF transform: z ~ G(0, 1) equiv. -log(-log(u)), u ~ U(0, 1).

    Args:
        mean (Optional[torch.Tensor], optional): Gumbel mean. Defaults to None.
        scale (Optional[torch.Tensor], optional): Gumbel scale. Defaults to None.
        fast (bool, optional): If True, will sample using log(-log(u)) where u ~ Uniform(eps, 1-eps).
                               Otherwise, samples via log(e) where e ~ Exponential(1). Defaults to True.
        eps (float, optional): Small constant for numerical stability in fast sampling. Defaults to 1e-10.

    Returns:
        torch.Tensor: Reparamterized samples from Gumbel distribution with given `mean` and `scale`.
    """
    if fast:
        gumbel = -torch.log(-torch.log(torch.empty_like(mean).uniform_(eps, 1.0 - eps)))
    else:
        gumbel = -torch.empty_like(mean).exponential_().log()

    return mean + scale * gumbel


@jit.script
def rsample_gumbel_softmax(
    logits: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
    return_argmax: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
):
    """Returns a sample from the Gumbel-Softmax distribution and optionally discretizes it [1, 2].

    As the softmax temperature τ approaches 0, samples from the Gumbel-Softmax distribution become one-hot
    and the Gumbel-Softmax distribution becomes identical to the categorical distribution Cat(logits).

    Args:
        logits (torch.Tensor): `[..., num_features]` unnormalized log probabilities
        tau (float): non-negative scalar temperature
        hard (bool): If True, the returned samples will be discretized as one-hot vectors,
                     but will be differentiated as if it is the soft sample in autograd.
        return_argmax (bool): If True, return the hard argmax indices instead of one_hot vectors. Defaults to False.
                              Has not effect when `hard=False`.
        dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
        Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
        If ``hard=True``, the returned samples will be one-hot, otherwise they will
        be probability distributions that sum to 1 across `dim`.

    Note:
        The main trick for `hard` is to do  `y_hard + (y_soft - y_soft.detach())`
        This achieves two things:
         1. makes the output value exactly `y_hard` and hence one-hot (since we add then subtract y_soft value)
         2. makes the (undefined) gradient of `y_hard` equal to y_soft gradient (since we strip all other gradients)
        The introduced gradient bias between `y_soft` and `y_hard` is reduced as `tau` is decreased towards zero.

    Examples:
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using pathwise gradient estimator:
        >>> rsample_gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" gradient estimator:
        >>> rsample_gumbel_softmax(logits, tau=1, hard=True)

    [1] https://arxiv.org/abs/1611.00712
    [2] https://arxiv.org/abs/1611.01144
    """
    gumbels = -torch.log(-torch.log(torch.empty_like(logits).uniform_(eps, 1.0 - eps)))  # ~Gumbel(0,1)
    logits_sampled = (logits + gumbels) / tau
    y_soft = logits_sampled.softmax(dim)

    if not hard:
        # Reparametrization trick with Gumbel Softmax (bias -> 0 as tau -> 0)
        return y_soft

    # Straight through estimator (always biased)
    index = y_soft.max(dim, keepdim=True)[1]
    if return_argmax:
        return index

    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return y_hard + (y_soft - y_soft.detach())


@jit.script
def rsample_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """
    Returns a sample from Logistic with specified mean and log scale.

    :param mu: a tensor containing the mean.
    :param log_scale: a tensor containing the log scale.
    :return: a reparameterized sample with the same size as the input mean and log scale.
    """
    u = torch.empty_like(mu).uniform_(eps, 1 - eps)  # uniform sample in the interval (eps, 1 - eps)
    sample = mu + torch.exp(log_scale) * (torch.log(u) - torch.log(1 - u))  # transform to logistic
    return sample


@jit.script
def rsample_discretized_logistic(mu: torch.Tensor, log_scale: torch.Tensor, eps: float = 1e-8):
    """Return a sample from a discretized logistic with values standardized to be in [-1, 1]

    This is done by sampling the corresponding continuous logistic and clamping values outside
    the interval to the endpoints.

    We do not further quantize the samples here.
    """
    return rsample_logistic(mu, log_scale, eps).clamp(-1, 1)


# @jit.script
def rsample_discretized_logistic_mixture(
    logit_probs: torch.Tensor,
    locs: torch.Tensor,
    log_scales: torch.Tensor,
    eps: float = 1e-5,
    rsample_categorical: bool = False,
    tau: float = 1.0,
):
    """Return a reparameterized sample from a given Discretized Logistic Mixture distribution.

    Code taken from PyTorch adaptation of original PixelCNN++ TensorFlow implementation:
    https://github.com/pclucas14/pixel-cnn-pp/blob/master/utils.py but does not include the channel specific conditional modelling.

    Args:
        logit_probs (torch.Tensor): (*, num_mix)
        locs (torch.Tensor): (*, D, num_mix)
        log_scales (torch.Tensor): (*, D, num_mix)
        num_mix (int): Number of mixture components
        eps (float): Bounds [eps, 1-eps] on the uniform rv used to sample the mixture coefficients and the logistic.
        tau (float): Temperature for Gumbel sampling

    Returns:
        torch.Tensor: Sample from the DLM `(*, D)`
    """
    # sample mixture indicator from categorical
    if rsample_categorical:
        argmax = rsample_gumbel_softmax(logit_probs, hard=True, tau=tau, return_argmax=True)
    else:
        gumbel = -torch.log(-torch.log(torch.empty_like(logit_probs).uniform_(eps, 1.0 - eps)))  # (*, num_mix)
        argmax = torch.argmax(logit_probs + gumbel, dim=-1, keepdim=True)  # (*, 1)

    # broadcast argmax onto mu and sd shapes (including D)
    argmax = argmax.expand(*argmax.shape[:-1], locs.size(-2)).unsqueeze(-1)  # (*, D, 1)

    # select component and remove mixture dimension
    locs = torch.gather(locs, index=argmax, dim=-1).squeeze(-1)
    log_scales = torch.gather(log_scales, index=argmax, dim=-1).squeeze(-1)

    # sample from logistic (we don't actually round to the nearest 8bit value)
    x = rsample_discretized_logistic(locs, log_scales)
    return x
