import math

from typing import Optional, Tuple

import torch
import torch.nn as nn

from torchtyping import TensorType

from blvm.utils.log_likelihoods import (
    gaussian_ll,
    gaussian_mixture_ll,
    categorical_ll,
    bernoulli_ll,
    discretized_logistic_ll,
    discretized_logistic_mixture_ll,
)
from blvm.utils.variational import (
    rsample_gaussian,
    rsample_gaussian_mixture,
    rsample_discretized_logistic,
    rsample_discretized_logistic_mixture,
)

from .convenience import AddConstant


class ConditionalDistribution(nn.Module):
    def __init__(self):
        super().__init__()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(*args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def sample(params):
        raise NotImplementedError()

    @staticmethod
    def rsample(params):
        raise NotImplementedError()

    @staticmethod
    def mode(params):
        raise NotImplementedError()

    def log_prob(self, x):
        raise NotImplementedError()


class IsotropicGaussianDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim, initial_sd: float = 1, epsilon: float = 1e-6):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.initial_sd = initial_sd
        self.epsilon = epsilon

        self.out_features = y_dim + 1

        self.params = nn.Linear(x_dim, y_dim + 1)

        if epsilon > 0:
            self.sd_activation = nn.Sequential(
                nn.Softplus(beta=math.log(2) / (initial_sd - epsilon)), AddConstant(epsilon)
            )
        else:
            self.sd_activation = nn.Softplus(beta=math.log(2) / (initial_sd - epsilon))

        self.reset_parameters()

    @staticmethod
    def get_distribution(params):
        return torch.distributions.Normal(loc=params[0], scale=params[1])

    @torch.no_grad()
    def sample(self, params):
        return rsample_gaussian(params[0], params[1])

    def rsample(self, params):
        return rsample_gaussian(params[0], params[1])

    def mode(self, params):
        return params[0]

    def log_prob(self, y, params, reduce_dim: Optional[int] = None):
        log_prob = gaussian_ll(y, params[0], params[1], epsilon=0)
        if reduce_dim is not None:
            return log_prob.sum(reduce_dim)
        return log_prob

    def forward(self, x):
        params = self.params(x)
        mu, log_sd = params[..., :-1], params[..., -1:]
        sd = self.sd_activation(log_sd)
        return mu, sd


@torch.jit.ignore
class DiagonalGaussianDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim, initial_sd: float = 1, epsilon: float = 1e-6):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.initial_sd = initial_sd
        self.epsilon = epsilon

        self.out_features = 2 * y_dim

        self.params = nn.Linear(x_dim, 2 * y_dim)

        if epsilon > 0:
            self.sd_activation = nn.Sequential(
                nn.Softplus(beta=math.log(2) / (initial_sd - epsilon)), AddConstant(epsilon)
            )
        else:
            self.sd_activation = nn.Softplus(beta=math.log(2) / (initial_sd - epsilon))

        self.reset_parameters()

    @staticmethod
    def get_distribution(params: Tuple[torch.Tensor, torch.Tensor]):
        return torch.distributions.Normal(loc=params[0], scale=params[1])

    @torch.no_grad()
    def sample(self, params: Tuple[torch.Tensor, torch.Tensor]):
        return rsample_gaussian(params[0], params[1])

    def rsample(self, params: Tuple[torch.Tensor, torch.Tensor]):
        return rsample_gaussian(params[0], params[1])

    def mode(self, params: Tuple[torch.Tensor, torch.Tensor]):
        return params[0].contiguous()

    def log_prob(self, y: torch.Tensor, params: Tuple[torch.Tensor, torch.Tensor], reduce_dim: Optional[int] = None):
        return gaussian_ll(y, params[0], params[1], epsilon=0, reduce_dim=reduce_dim)

    def forward(self, x: torch.Tensor):
        params = self.params(x)
        mu, log_sd = params.chunk(2, dim=-1)
        sd = self.sd_activation(log_sd)
        return mu, sd


class DiagonalGaussianMixtureDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim, num_mix: int, initial_sd: float = 1, epsilon: float = 1e-6):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_mix = num_mix
        self.initial_sd = initial_sd
        self.epsilon = epsilon

        self.out_features = num_mix * (2 * y_dim + 1)

        self.params = nn.Linear(x_dim, self.out_features)

        if epsilon > 0:
            self.sd_activation = nn.Sequential(nn.Softplus(beta=math.log(2) / (initial_sd)), AddConstant(epsilon))
        else:
            self.sd_activation = nn.Softplus(beta=math.log(2) / (initial_sd - epsilon))

        self.reset_parameters()

    @torch.no_grad()
    def sample(self, params: Tuple[torch.Tensor, torch.Tensor]):
        return rsample_gaussian_mixture(params[0], params[1], params[2])

    def rsample(self, params: Tuple[torch.Tensor, torch.Tensor]):
        return rsample_gaussian_mixture(params[0], params[1], params[2])

    def mode(self, params: Tuple[torch.Tensor, torch.Tensor]):
        mode_component = params[0].argmax(-1, keepdim=True).unsqueeze(-2)
        mode = torch.gather(params[1], index=mode_component, dim=-1).squeeze(-1)
        return mode

    def log_prob(self, y: torch.Tensor, params: Tuple[torch.Tensor, torch.Tensor], reduce_dim: int = -1):
        return gaussian_mixture_ll(y, params[0], params[1], params[2], epsilon=0, reduce_dim=reduce_dim)

    def forward(self, x: torch.Tensor):
        """Returns parameters of the GMM.

        Args:
            x (torch.Tensor): [description]

        Returns:
            tuple: log-coefficients (*, num_mix), means (*, D, num_mix), standard deviations (*, D, num_mix)
        """
        params = self.params(x)
        logit_probs = params[..., :self.num_mix]  # (*, num_mix)
        mu_log_sd = params[..., self.num_mix:].view(*params.shape[:-1], self.y_dim, 2 * self.num_mix)  # (*, D, num_mix)
        mu, log_sd = mu_log_sd.chunk(2, dim=-1)
        sd = self.sd_activation(log_sd)
        return logit_probs, mu, sd


class CategoricalDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim):
        """Parameterizes a Gaussian distribution with diagonal covariance"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.logits = nn.Linear(x_dim, y_dim)

        self.reset_parameters()

    @staticmethod
    def get_distribution(logits):
        return torch.distributions.Categorical(logits=logits)

    @staticmethod
    def sample(logits):
        return torch.distributions.Categorical(logits=logits).sample()

    @staticmethod
    def mode(logits, dim: int = -1):
        return torch.argmax(logits, dim=dim)

    def log_prob(self, y, logits, reduce_dim: Optional[int] = -1):
        return categorical_ll(y, logits, reduce_dim=reduce_dim)

    def forward(self, x):
        return self.logits(x)


class BernoulliDense(ConditionalDistribution):
    def __init__(self, x_dim, y_dim):
        """Parameterizes a Bernoulli distribution"""
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim

        self.logits = nn.Linear(x_dim, y_dim)

        self.reset_parameters()

    @staticmethod
    def get_distribution(logits):
        return torch.distributions.Bernoulli(logits=logits)

    @staticmethod
    def sample(logits):
        return torch.distributions.Bernoulli(logits=logits).sample()

    def mode(self, logits):
        return torch.argmax(logits, dim=self.reduce_dim)

    def log_prob(self, y, logits, reduce_dim: Optional[int] = None):
        return bernoulli_ll(y, logits).sum(reduce_dim)

    def forward(self, x):
        return self.logits(x)


class DiscretizedLogisticDense(ConditionalDistribution):
    def __init__(self, x_dim: int, y_dim: int, num_bins: int = 256, log_epsilon: float = -7.0):
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_bins = num_bins
        self.log_epsilon = log_epsilon

        self.out_features = y_dim * 2

        self.params = nn.Linear(x_dim, self.out_features)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def rsample(params):
        return rsample_discretized_logistic(params[0], params[1])

    @staticmethod
    @torch.no_grad()
    def sample(params):
        return rsample_discretized_logistic(params[0], params[1])

    def mode(self, params):
        return params[0]

    def log_prob(self, y, params, reduce_dim: Optional[int] = None):
        """Compute log-likelihood. Inputs are assumed to be [-1, 1]"""
        log_prob = discretized_logistic_ll(y, params[0], params[1], num_bins=self.num_bins, reduce_dim=reduce_dim)
        return log_prob

    def forward(self, x):
        params = self.params(x)  # (*, D, 2)
        mu, log_scale = params.chunk(2, dim=-1)
        log_scale = log_scale.clamp(min=self.log_epsilon)
        return mu, log_scale


class DiscretizedLogisticMixtureDense(ConditionalDistribution):
    def __init__(self, x_dim: int, y_dim: int, num_mix: int = 10, num_bins: int = 256, log_epsilon: float = -7.0):
        """Discretized Logistic Mixture distribution.

        The distribution has the following params:

        - Mean value per mixture: `num_mix`.
        - Log-scale per mixture: `num_mix`.
        - Mixture coefficient per mixture: `num_mix`.

        This yields a total of `3 * num_mix` params.
        This is different to the Discretized Mixture of Logistics used the PixelCNN++ paper which is tailored
        for RGB images and treats the channel dimension in a speciail way. There are no such special dimensions here.

        Assumes input data to be originally int (0, ..., num_bins) and then rescaled to num_bins
        discrete values in [-1, 1].

        Mean is not implemented for now.

        Args:
            x_dim (int): Number of channels in the input
            y_dim (int): Number of channels in the output
            num_mix (int, optional): Number of components. Defaults to 10.
            num_bins (int, optional): Number of quantization bins. Defaults to 256 (8 bit).
        """
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.log_epsilon = log_epsilon

        self.out_features = num_mix * (2 * y_dim + 1)

        self.params = nn.Linear(x_dim, self.out_features)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    @staticmethod
    def get_distribution(params):
        raise NotImplementedError("Discretized mixture of logistics does not have a Distribution object (yet)")

    def rsample(self, params):
        return rsample_discretized_logistic_mixture(params[0], params[1], params[2])

    @torch.no_grad()
    def sample(self, params):
        return rsample_discretized_logistic_mixture(params[0], params[1], params[2])

    def mode(self, params):
        """Return the mode as the mode (i.e. mean) of the most probably component"""
        mode_component = params[0].argmax(-1, keepdim=True).unsqueeze(-2)
        mode = torch.gather(params[1], index=mode_component, dim=-1).squeeze(-1)
        mode = mode.contiguous()
        return mode

    def log_prob(self, y, params, reduce_dim: int = -1):
        """Compute log-likelihood. Inputs are assumed to be [-1, 1]"""
        return discretized_logistic_mixture_ll(
            y,
            params[0],
            params[1],
            params[2],
            num_bins=self.num_bins,
            reduce_dim=reduce_dim,
        )

    def forward(self, x: TensorType["B", "T", "D"]):
        params = self.params(x)  # (*, D x 3 x self.num_mix)
        logit_probs = params[..., :self.num_mix]  # (*, num_mix)
        locs_log_scales = params[..., self.num_mix:].view(*params.shape[:-1], self.y_dim, 2 * self.num_mix)  # (*, D, num_mix)
        locs, log_scales = locs_log_scales.chunk(2, dim=-1)
        log_scales = log_scales.clamp(min=self.log_epsilon)
        return logit_probs, locs, log_scales
