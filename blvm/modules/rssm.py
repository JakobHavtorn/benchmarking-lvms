from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
import torch.jit as jit

from blvm.modules.distributions import DiagonalGaussianDense
from blvm.utils.variational import precision_weighted_gaussian


RSSMOutputs = namedtuple(
    typename="RSSMOutputs",
    field_names=["z", "enc_mu", "enc_sd", "prior_mu", "prior_sd"],
)


class RSSMCell(nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        c_dim: int,
        e_dim: int,
        residual_posterior: bool = False,
        precision_posterior: bool = False,
    ):
        """Recurrent State Space Model cell

        Args:
            z_dim (int): Dimensionality of the internal stochastic state space.
            h_dim (int): Dimensionality of the internal deterministic state space.
            c_dim (int): Dimensionality of the external temporal context.
            e_dim (int): Dimensionality of the external input embedding space used for inferred posterior distribution.
        """
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.c_dim = c_dim
        self.residual_posterior = residual_posterior
        self.precision_posterior = precision_posterior

        self.gru_in = nn.Sequential(nn.Linear(z_dim + c_dim, h_dim), nn.ReLU())
        self.gru_cell = nn.GRUCell(h_dim, h_dim)

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            DiagonalGaussianDense(h_dim, z_dim),
        )

        self.posterior = nn.Sequential(
            nn.Linear(h_dim + e_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            DiagonalGaussianDense(h_dim, z_dim),
        )

    @jit.export
    def get_initial_state(self, batch_size: int, device: str = ""):
        device = torch.device(device) if device != "" else self.prior[0].weight.device
        return (torch.zeros(batch_size, self.z_dim, device=device), torch.zeros(batch_size, self.h_dim, device=device))

    @jit.export
    def get_empty_context(self, batch_size: int, device: str = ""):
        device = torch.device(device) if device != "" else self.prior[0].weight.device
        return torch.empty(batch_size, 0, device=device)

    @jit.export
    def forward(
        self,
        enc_inputs: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        use_mode: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RSSMOutputs]:
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        enc_mu, enc_sd = self.posterior(torch.cat([h_new, enc_inputs], dim=-1))

        prior_mu, prior_sd = self.prior(h_new)

        if self.residual_posterior:
            enc_mu = enc_mu + prior_mu
        elif self.precision_posterior:
            enc_mu, enc_sd = precision_weighted_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)

        z_new = self.posterior[-1].rsample((enc_mu, enc_sd)) if not use_mode else enc_mu

        distributions = RSSMOutputs(z=z_new, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd)

        return (z_new, h_new), distributions

    @jit.export
    def generate(
        self,
        state: Tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        use_mode: bool = False,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], RSSMOutputs]:
        z, h = state

        gru_in = self.gru_in(torch.cat([z, context], dim=-1))
        h_new = self.gru_cell(gru_in, h)

        prior_mu, prior_sd = self.prior(h_new)
        z_new = self.prior[-1].rsample((prior_mu, prior_sd)) if not use_mode else prior_mu

        distributions = RSSMOutputs(z=z_new, prior_mu=prior_mu, prior_sd=prior_sd, enc_mu=torch.empty((0,)), enc_sd=torch.empty((0,)))

        return (z_new, h_new), distributions
