import math

from collections import namedtuple
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.jit as jit

from torchtyping import TensorType
from blvm.data.transforms import StackTensor

from blvm.evaluation import LossMetric, LLMetric, KLMetric, BitsPerDimMetric, LatestMeanMetric
from blvm.models import BaseModel
from blvm.modules.convenience import View
from blvm.modules.distributions import (
    DiagonalGaussianMixtureDense,
    DiscretizedLogisticMixtureDense,
    DiagonalGaussianDense,
    BernoulliDense,
    CategoricalDense,
)
from blvm.modules.dropout import WordDropout
from blvm.utils.operations import sequence_mask
from blvm.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian


VRNNOutputs = namedtuple(
    typename="VRNNCellOutputs",
    field_names=["h", "z", "enc_mu", "enc_sd", "prior_mu", "prior_sd", "phi_z"],
)


class VRNNCell(nn.Module):
    def __init__(
        self, x_dim: int, h_dim: int, z_dim: int, r_dim: Optional[int] = None, condition_h_on_x: bool = True, residual_posterior: bool = False
    ):
        """Variational Recurrent Neural Network (VRNN) cell from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

        Args:
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size
            condition_h_on_x (bool): If True, condition h on x observation in inference and generation.

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()
        
        r_dim = r_dim if r_dim else 2 * h_dim

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.condition_h_on_x = condition_h_on_x
        self.residual_posterior = residual_posterior

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )

        self.prior = nn.Sequential(
            nn.Linear(r_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            DiagonalGaussianDense(h_dim, z_dim),
        )

        self.posterior = nn.Sequential(
            nn.Linear(x_dim + r_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            DiagonalGaussianDense(h_dim, z_dim),
        )
        gru_in_dim = x_dim + h_dim if self.condition_h_on_x else h_dim
        self.gru_cell = nn.GRUCell(gru_in_dim, r_dim)

        self.reset_parameters()

    @jit.ignore
    def reset_parameters(self) -> None:
        init.orthogonal_(self.gru_cell.weight_hh)

    @jit.export
    def get_initial_state(self, batch_size: int, device: str = ""):
        device = torch.device(device) if device != "" else self.prior[0].weight.device
        h0 = torch.zeros(batch_size, self.r_dim, device=device)
        return h0

    @jit.export
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, VRNNOutputs]:
        """Forward pass

        Args:
            x (Tensor): (B, x_dim)
            h (Tensor): (B, h_dim)

        Returns:
            tuple: Next h, features for z, posterior mean and sd, prior mean and sd, extra outputs
        """
        # prior p(z)
        prior_mu, prior_sd = self.prior(h)

        # encoder q(z|x)
        enc_mu, enc_sd = self.posterior(torch.cat([h, x], -1))

        if self.residual_posterior:
            enc_mu = enc_mu + prior_mu

        # sampling and reparameterization
        z = rsample_gaussian(enc_mu, enc_sd)

        # z features
        phi_z = self.phi_z(z)

        # gru cell
        if self.condition_h_on_x:
            h = self.gru_cell(torch.cat([x, phi_z], -1), h)
        else:
            h = self.gru_cell(phi_z, h)

        outputs = VRNNOutputs(h=h, z=z, enc_mu=enc_mu, enc_sd=enc_sd, prior_mu=prior_mu, prior_sd=prior_sd, phi_z=phi_z)
        return h, outputs

    @jit.export
    def generate(self, x: torch.Tensor, h: torch.Tensor, use_mode: bool = False):
        # prior p(z)
        prior_mu, prior_sd = self.prior(h)

        # sampling and reparameterization
        if use_mode:
            z = prior_mu
        else:
            z = rsample_gaussian(prior_mu, prior_sd)

        # z features
        phi_z = self.phi_z(z)

        # gru cell
        if self.condition_h_on_x:
            h = self.gru_cell(torch.cat([x, phi_z], -1), h)
        else:
            h = self.gru_cell(phi_z, h)

        outputs = VRNNOutputs(h=h, z=z, enc_mu=prior_mu, enc_sd=prior_sd, prior_mu=prior_mu, prior_sd=prior_sd, phi_z=phi_z)
        return h, outputs


class VRNN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        likelihood: nn.Module,
        x_dim: int,
        h_dim: int,
        z_dim: int,
        r_dim: Optional[int] = None,
        decoder: nn.Module = None,
        residual_posterior: bool = False,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
        dropout: float = 0,
    ):
        """Variational Recurrent Neural Network (VRNN) from [1].

        Uses unimodal isotropic gaussian distributions for inference, prior, and generative models.

            ┌───┐       ┌───┐       ┌───┐          ┌───┐       ┌───┐       ┌───┐
            │h_1├─┬────►│h_2├─┬────►│h_3│          │h_1├─┬────►│h_2├─┬────►│h_3│
            └─┬─┘ │     └─┬─┘ │     └─┬─┘          └─┬─┘ │     └─┬─┘ │     └─┬─┘
              │   │       │   │       │              │   │       │   │       │
              │   │       │   │       │              │   │   ┌───┤   │   ┌───┤
              │   │       │   │       │              │   │   │   │   │   │   │
              ▼   │       ▼   │       ▼              ▼   │   │   ▼   │   │   ▼
            ┌───┐ │     ┌───┐ │     ┌───┐          ┌───┐ │   │ ┌───┐ │   │ ┌───┐
            │z_1├─┤     │z_2├─┤     │z_3│          │z_1├─┤   │ │z_2├─┤   │ │z_3│
            └───┘ │     └───┘ │     └───┘          └─┬─┘ │   │ └─┬─┘ │   │ └─┬─┘
              ▲   │       ▲   │       ▲              │   │   │   │   │   │   │
              │   │       │   │       │              │   │   │   │   │   │   │
              │   │       │   │       │              │   │   │   │   │   │   │
              │   │       │   │       │              ▼   │   │   ▼   │   │   ▼
            ┌─┴─┐ │     ┌─┴─┐ │     ┌─┴─┐          ┌───┐ │   │ ┌───┐ │   │ ┌───┐
            │x_1├─┘     │x_2├─┘     │x_3│          │x_1├─┘   └►│x_2├─┘   └►│x_3│
            └───┘       └───┘       └───┘          └───┘       └───┘       └───┘

                   INFERENCE MODEL                       GENERATIVE MODEL

        Args:
            encoder (nn.Module): Input transformation
            x_dim (int): Input space size
            h_dim (int): Hidden space (GRU) size
            z_dim (int): Stochastic latent variable size
            o_dim (int): Output space size
            condition_h_on_x (bool): If True, condition h on x in inference and generation (parameter sharing).
            condition_x_on_h (bool): If True, condition x on h in generation.

        [1] https://arxiv.org/abs/1506.02216
        """
        super().__init__()
        r_dim = r_dim if r_dim else 2 * h_dim

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.residual_posterior = residual_posterior
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h

        self.encoder = encoder
        self.likelihood = likelihood
        if decoder is None:
            decoder_in_dim = h_dim + r_dim if condition_x_on_h else h_dim
            self.decoder = nn.Sequential(
                nn.Linear(decoder_in_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.ReLU(),
            )
        else:
            self.decoder = decoder

        self.vrnn_cell = VRNNCell(
            x_dim=x_dim,
            h_dim=h_dim,
            z_dim=z_dim,
            condition_h_on_x=condition_h_on_x,
            residual_posterior=residual_posterior,
        )
        self.vrnn_cell = jit.script(self.vrnn_cell)

        self.likelihood = likelihood
        self.dropout = nn.Dropout(dropout) if dropout else None

    def compute_elbo(
        self,
        y: TensorType["B", "T"],
        parameters: TensorType["B", "T", "D"],
        kld_twise: TensorType["B", "T", "latent_size"],
        x_sl: TensorType["B", int],
        stride: int,
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        seq_mask = sequence_mask(x_sl, dtype=float, device=y.device)

        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask  # (B, T)
        log_prob = log_prob_twise.flatten(1).sum(1)  # (B,)

        seq_mask_kl = seq_mask[:, ::stride].unsqueeze(-1)
        kld = (kld_twise * seq_mask_kl).sum((1, 2))  # (B,)
        elbo = log_prob - kld  # (B,)

        kld_twise_fn = discount_free_nats(kld_twise, free_nats, shared_dims=-1)
        kld = (kld_twise_fn * seq_mask_kl).sum((1, 2))  # (B,)
        loss = -(log_prob - beta * kld).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kld, seq_mask

    def forward(
        self,
        x: TensorType["B", "T", "x_dim"],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):

        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # target
        y = x.clone().detach()

        # x features
        encoding = self.encoder(x)  # (B, T, D)
        stride = math.ceil(x.size(1) / encoding.size(1))
        encoding = encoding.unbind(1)  # [(B, D)]

        # initial h
        h = self.vrnn_cell.get_initial_state(x.size(0)) if h0 is None else h0
        all_h = [h]

        all_outputs = []
        for t in range(len(encoding)):
            h, outputs = self.vrnn_cell(encoding[t], h)
            all_outputs.append(outputs)

        all_h.extend([output.h for output in all_outputs])
        all_h.pop()  # Include initial and not last

        all_z = [output.z for output in all_outputs]
        all_phi_z = [output.phi_z for output in all_outputs]
        all_enc_mu = [output.enc_mu for output in all_outputs]
        all_enc_sd = [output.enc_sd for output in all_outputs]
        all_prior_mu = [output.prior_mu for output in all_outputs]
        all_prior_sd = [output.prior_sd for output in all_outputs]

        # output distribution
        phi_z = torch.stack(all_phi_z, dim=1)
        if self.condition_x_on_h:
            h = torch.stack(all_h, dim=1)
            dec = self.decoder(torch.cat([phi_z, h], -1))
        else:
            dec = self.decoder(phi_z)

        dec = dec[:, :x_sl.max(), :]  # remove right padding (if any)
        dec = self.dropout(dec) if self.dropout is not None else dec

        parameters = self.likelihood(dec)  # (B, T, D)
        reconstruction = self.likelihood.sample(parameters)
        reconstruction_mode = self.likelihood.mode(parameters)

        # kl divergence, elbo and loss
        enc_mu = torch.stack(all_enc_mu, dim=1)
        enc_sd = torch.stack(all_enc_sd, dim=1)
        prior_mu = torch.stack(all_prior_mu, dim=1)
        prior_sd = torch.stack(all_prior_sd, dim=1)
        kld = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)

        loss, elbo, log_prob, kl, seq_mask = self.compute_elbo(y, parameters, kld, x_sl, stride, beta, free_nats)

        z = torch.stack(all_z, dim=1)
        z_sl = (x_sl / stride).ceil().int()
        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            LLMetric(elbo, name="elbo"),
            LLMetric(log_prob, name="rec"),
            KLMetric(kl),
            KLMetric(kl / math.log(2), name="kl (bpt)", reduce_by=x_sl),
            BitsPerDimMetric(elbo, reduce_by=x_sl),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]
        outputs = SimpleNamespace(
            elbo=elbo,
            log_prob=log_prob,
            kl=kl,
            y=y,
            seq_mask=seq_mask,
            z=z,
            z_sl=z_sl,
            reconstructions=reconstruction,
            reconstructions_mode=reconstruction_mode,
            reconstructions_parameters=parameters,
            h_n=all_h[-1],
        )
        return loss, metrics, outputs

    def generate(
        self,
        x: TensorType["B", "x_dim"],
        h0: TensorType["B", "h_dim"] = None,
        n_samples: int = 1,
        max_timesteps: int = 100,
        stop_value: float = None,
        use_mode: bool = False,
    ):

        if x.size(0) > 1:
            assert x.size(0) == n_samples
        else:
            x = x.repeat(n_samples, *[1] * (x.ndim - 1))  # Repeat along batch

        # TODO If multiple timesteps in x, run a forward pass to get conditional initial h (assert h is None)
        all_outputs = []
        all_x = [x]
        x_sl = torch.ones(n_samples, dtype=torch.int)

        # initial h
        h = self.vrnn_cell.get_initial_state(x.size(0)) if h0 is None else h0

        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop

        while not all_ended and t < max_timesteps:
            encoding = self.encoder(x)

            h, outputs = self.vrnn_cell.generate(encoding[:, 0, :], h)

            if self.condition_x_on_h:
                dec = self.decoder(torch.cat([outputs.phi_z, h], -1))
            else:
                dec = self.decoder(outputs.phi_z)
            parameters = self.likelihood(dec)  # (B, T, D)

            if use_mode:
                x = self.likelihood.mode(parameters)
            else:
                x = self.likelihood.sample(parameters)

            all_x.append(x)
            all_outputs.append(outputs)

            # Update sequence length
            x_sl += seq_active
            seq_ending = x == stop_value  # (,), (B,) or (B, D*)
            if isinstance(seq_ending, torch.Tensor):
                seq_ending = seq_ending.all(*list(range(1, seq_ending.ndim))) if seq_ending.ndim > 1 else seq_ending
                seq_ending = seq_ending.to(int).cpu()
            else:
                seq_ending = int(seq_ending)
            seq_active *= 1 - seq_ending

            # Update loop conditions
            t += 1
            all_ended = torch.all(1 - seq_active).item()

        x = torch.cat(all_x, dim=-1)  # (B, D, T)
        x = x.permute(0, 2, 1)  # (B, T, D)

        outputs = SimpleNamespace()
        return (x, x_sl), outputs


class VRNNAudio(BaseModel):
    def __init__(
        self,
        likelihood: Union[str, nn.Module],
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
        residual_posterior: bool = False,
        condition_h_on_x: bool = True,
        condition_x_on_h: bool = True,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        """A VRNN for modelling audio waveforms."""
        super().__init__()
        self.likelihood = likelihood
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.residual_posterior = residual_posterior
        self.condition_h_on_x = condition_h_on_x
        self.condition_x_on_h = condition_x_on_h
        self.num_mix = num_mix
        self.num_bins = num_bins

        if isinstance(likelihood, str):
            if likelihood == "DMoL":
                likelihood_module = DiscretizedLogisticMixtureDense(
                    x_dim=2 * num_mix + num_mix,
                    y_dim=1,
                    num_mix=10,
                    num_bins=2 ** 16,
                )
            elif likelihood == "GMM":
                likelihood_module = DiagonalGaussianMixtureDense(
                    x_dim=2 * num_mix + num_mix,
                    y_dim=1,
                    num_mix=num_mix,
                    initial_sd=1,
                    epsilon=1e-4,
                )
            elif likelihood == "Gaussian":
                likelihood_module = DiagonalGaussianDense(
                    x_dim=2,
                    y_dim=1,
                    epsilon=1e-4,
                )
            else:
                raise ValueError(f"Unknown likelihood type {likelihood}")

        encoder = nn.Sequential(
            View(-1),  # (B, T, 1) to (B, T)
            StackTensor(input_size, dim=1),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )
        decoder = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size * likelihood_module.out_features),  # (B, T/S, S*D)
            nn.LeakyReLU(),
            View(-1, likelihood_module.out_features),  # (B, T, D)
        )
        self.vrnn = VRNN(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood_module,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            residual_posterior=residual_posterior,
            condition_h_on_x=condition_h_on_x,
            condition_x_on_h=condition_x_on_h,
        )

    def forward(
        self,
        x: TensorType["B", "T", "D", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        h0: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.vrnn(x, x_sl, beta, free_nats, h0)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        x = torch.zeros(n_samples, self.input_size, 1, device=self.device) if x is None else x
        return self.vrnn.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            stop_value=None,
            use_mode=use_mode,
            x=x,
            h0=h0,
        )
