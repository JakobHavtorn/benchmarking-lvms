import math

from types import SimpleNamespace
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.init as init

from torchtyping import TensorType
from blvm.data.transforms import StackTensor

from blvm.evaluation.metrics import BitsPerDimMetric, KLMetric, LLMetric, LatestMeanMetric, LossMetric, PerplexityMetric
from blvm.models import BaseModel
from blvm.models.clockwork_vae.convolutional_coders import ConvCoder1d
from blvm.modules import (
    CategoricalDense,
    DiagonalGaussianDense,
    DiscretizedLogisticMixtureDense,
)
from blvm.modules.convenience import View
from blvm.modules.distributions import DiagonalGaussianDense, DiagonalGaussianMixtureDense
from blvm.utils.padding import get_modulo_length
from blvm.utils.variational import discount_free_nats, kl_divergence_gaussian, rsample_gaussian
from blvm.utils.operations import sequence_mask, reverse_sequences, split_sequence


class SRNN(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        likelihood,
        x_dim,
        h_dim,
        z_dim,
        r_dim: Optional[int] = None,
        gated_stochastic_transfer: bool = False,
        use_phi_z: bool = False,
        dropout: float = 0,
        num_layers: int = 1,
        residual_posterior: bool = False,
        smoothing: bool = True,
    ):
        """Stochastic Recurrent Neural Network from [1]

        If `gated_stochastic_transfer` is True, then we use a GRU instead of Elman RNN for the stochastic transfer.

        [1] https://arxiv.org/abs/1605.07571
        """
        super(SRNN, self).__init__()

        r_dim = 2 * h_dim if r_dim is None else r_dim

        self.encoder = encoder
        self.decoder = decoder
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.r_dim = r_dim
        self.use_phi_z = use_phi_z  # Make slightly more similar to VRNN by transforming z before input to GRU
        self.gated_stochastic_transfer = gated_stochastic_transfer
        self.dropout = dropout
        self.num_layers = num_layers
        self.residual_posterior = residual_posterior
        self.smoothing = smoothing

        if use_phi_z:
            self.phi_z = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.LeakyReLU(),
                nn.Linear(h_dim, h_dim),
                nn.LeakyReLU(),
                nn.Linear(h_dim, h_dim),
                nn.LeakyReLU(),
                nn.Linear(h_dim, h_dim),
                nn.LeakyReLU(),
            )
            phi_z_dim = h_dim
        else:
            self.phi_z = None
            phi_z_dim = z_dim

        if gated_stochastic_transfer:
            in_dim_q_p = r_dim
            in_dim_gru = r_dim + phi_z_dim
        else:
            in_dim_q_p = r_dim + phi_z_dim
            in_dim_gru = None

        # encoder  x/u to z, input to latent variable, inference model
        self.posterior = nn.Sequential(
            nn.Linear(in_dim_q_p, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            DiagonalGaussianDense(h_dim, z_dim),
        )

        # prior transition of zt-1 to zt
        self.prior = nn.Sequential(
            nn.Linear(in_dim_q_p, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            nn.Linear(h_dim, h_dim),
            nn.LeakyReLU(),
            DiagonalGaussianDense(h_dim, z_dim),
        )

        self.d_forward_recurrent = nn.GRU(x_dim, r_dim, num_layers)

        if smoothing:
            self.a_backward_recurrent = nn.GRU(x_dim + r_dim, r_dim, num_layers)
        else:
            self.a_mlp = nn.Sequential(
                nn.Linear(x_dim + r_dim, r_dim), nn.LeakyReLU(), nn.Linear(r_dim, r_dim), nn.LeakyReLU()
            )

        if gated_stochastic_transfer:
            self.gru_cell = nn.GRUCell(in_dim_gru, r_dim)

        self.dropout = nn.Dropout(dropout) if dropout else None
        self.likelihood = likelihood

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i_layer in range(self.num_layers):
            weight_name = f"weight_hh_l{i_layer}"
            init.orthogonal_(getattr(self.d_forward_recurrent, weight_name))
            if self.smoothing:
                init.orthogonal_(getattr(self.a_backward_recurrent, weight_name))

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
        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask
        log_prob = log_prob_twise.flatten(1).sum(1)  # (B,)

        seq_mask_kl = seq_mask[:, ::stride].unsqueeze(-1)
        kld = (kld_twise * seq_mask_kl).sum((1, 2))  # (B,)
        elbo = log_prob - kld  # (B,)

        kld_twise_fn = discount_free_nats(kld_twise, free_nats, shared_dims=-1)
        kld_fn = (kld_twise_fn * seq_mask_kl).sum((1, 2))  # (B,)
        loss = -(log_prob - beta * kld_fn).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kld, seq_mask

    def forward(
        self,
        x: TensorType["B", "T"],
        x_sl: TensorType["B", int],
        u: TensorType["B", "T"] = None,
        d_0: TensorType["num_layers", "B", "h_dim"] = None,
        a_0: TensorType["num_layers", "B", "h_dim"] = None,
        z_0: TensorType["B", "z_dim"] = None,
        h_p_0: TensorType["B", "h_dim"] = None,
        h_q_0: TensorType["B", "h_dim"] = None,
        beta: float = 1,
        free_nats: float = 0,
    ):
        batch_size = x.size(0)
        device = x.device

        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # target
        y = x.clone().detach()

        # x features
        x_encoding = self.encoder(x)

        stride = math.ceil(x.shape[1] / x_encoding.shape[1])
        x_sl_strided = (x_sl / stride).ceil().int()
        x_encoding = x_encoding.permute(1, 0, 2)  # (T, B, D)

        # u_features
        u_encoding = torch.cat([torch.zeros_like(x_encoding[0:1]), x_encoding[:-1, ...]], dim=0) if u is None else u

        # u_t to d_t
        d_0 = torch.zeros(self.num_layers, batch_size, self.r_dim, device=device) if d_0 is None else d_0
        d, d_n = self.d_forward_recurrent(u_encoding, d_0)
        d = torch.cat([d_0, d[:-1, ...]], dim=0)  # Pop last hidden, prepend initial

        # x_t and d_t to a_t
        concat_h_t_x_t = torch.cat([x_encoding, d], dim=-1)

        if self.smoothing:
            concat_h_t_x_t = reverse_sequences(concat_h_t_x_t, x_sl_strided)
            a_0 = torch.zeros(self.num_layers, batch_size, self.r_dim, device=device) if a_0 is None else a_0
            a, a_n = self.a_backward_recurrent(concat_h_t_x_t, a_0)
            a = reverse_sequences(a, x_sl_strided)  # reverse back again (index 0 == time 0)
        else:
            a = self.a_mlp(concat_h_t_x_t)
            a_n = None

        # prepare for iteration
        all_enc_mu, all_enc_sd = [], []
        all_prior_mu, all_prior_sd = [], []
        z_t_sampled = []

        d = d.permute(1, 0, 2)  # (T, B, D) to (B, T, D)
        a = a.permute(1, 0, 2)

        z_t = torch.zeros(batch_size, self.z_dim, device=device) if z_0 is None else z_0
        if self.gated_stochastic_transfer:
            h_p = torch.zeros(batch_size, self.r_dim, device=device) if h_p_0 is None else h_p_0
            h_q = torch.zeros(batch_size, self.r_dim, device=device) if h_q_0 is None else h_q_0

        for d_t, a_t in zip(d.unbind(1), a.unbind(1)):

            if self.use_phi_z:
                z_t = self.phi_z(z_t)

            if self.gated_stochastic_transfer:
                h_p = self.gru_cell(torch.cat([d_t, z_t], dim=-1), h_p)
                h_q = self.gru_cell(torch.cat([a_t, z_t], dim=-1), h_q)
            else:
                h_p = torch.cat([d_t, z_t], dim=-1)  # Elman RNN but conditioned on GRU state
                h_q = torch.cat([a_t, z_t], dim=-1)

            # prior conditioned on d_t and z_{t-1}
            prior_mu_t, prior_sd_t = self.prior(h_p)

            # encoder conditioned on a_t and z_{t-1}
            enc_mu_t, enc_sd_t = self.posterior(h_q)

            # residual parameterization of posterior
            if self.residual_posterior:
                enc_mu_t = enc_mu_t + prior_mu_t

            # sampling and reparameterization
            z_t = rsample_gaussian(enc_mu_t, enc_sd_t)

            all_prior_mu.append(prior_mu_t)
            all_prior_sd.append(prior_sd_t)
            all_enc_mu.append(enc_mu_t)
            all_enc_sd.append(enc_sd_t)
            z_t_sampled.append(z_t)

        # decoder emission (generative model)
        z = torch.stack(z_t_sampled, dim=1)
        dec = self.decoder(torch.cat([z, d], dim=-1))
        dec = dec[:, : x_sl.max(), :]  # remove right padding (if any)

        dec = self.dropout(dec) if self.dropout is not None else dec

        parameters = self.likelihood(dec)  # (B, T, D)
        reconstructions = self.likelihood.sample(parameters)
        reconstructions_mode = self.likelihood.mode(parameters)

        enc_mu = torch.stack(all_enc_mu, dim=1)
        enc_sd = torch.stack(all_enc_sd, dim=1)
        prior_mu = torch.stack(all_prior_mu, dim=1)
        prior_sd = torch.stack(all_prior_sd, dim=1)
        kld = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)

        loss, elbo, log_prob, kl, seq_mask = self.compute_elbo(y, parameters, kld, x_sl, stride, beta, free_nats)

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
            parameters=parameters,
            seq_mask=seq_mask,
            z=z,
            z_sl=x_sl_strided,
            d_n=d_n,
            a_n=a_n,
            z_n=z_t_sampled[-1],
            h_p_n=h_p,
            h_q_n=h_q,
            reconstructions=reconstructions,
            reconstructions_mode=reconstructions_mode,
            reconstructions_parameters=parameters,
        )
        return loss, metrics, outputs

    def generate(
        self,
        x: TensorType["B", "T", "x_dim"],
        u: TensorType["B", "T", "x_dim"] = None,
        d_0: TensorType["num_layers", "B", "h_dim"] = None,
        a_0: TensorType["num_layers", "B", "h_dim"] = None,
        z_0: TensorType["B", "z_dim"] = None,
        h_p_0: TensorType["B", "h_dim"] = None,
        n_samples: int = 1,
        max_timesteps: int = 100,
        stop_value: float = None,
        use_mode: bool = False,
    ):
        # device
        device = self.d_forward_recurrent.weight_hh_l0.device

        if x.size(1) > 1:
            # conditional generation
            x_sl = torch.full_like(x, fill_value=x.size(1) - 1)
            u_in = u[:, : x.size(1) - 1, :] if u is not None else u
            _, _, outputs = self.forward(x[:, :-1, :], x_sl, u_in, d_0=d_0, a_0=a_0, z_0=z_0)
            d_t = outputs.d
            z_t = outputs.z
            x = x[:, -1, :]
            u = u[:, : x.size(1) - 1, :]
        else:
            # unconditional generation
            x_sl = torch.zeros(n_samples)
            d_t = torch.zeros(self.num_layers, n_samples, self.r_dim, device=device) if d_0 is None else d_0
            z_t = torch.zeros(n_samples, self.z_dim, device=device) if z_0 is None else z_0
            h_p = torch.zeros(n_samples, self.r_dim, device=device) if h_p_0 is None else h_p_0

        all_x = []
        all_z = []

        seq_active = torch.ones(n_samples, dtype=torch.int)
        all_ended, t = False, 0  # Used to condition while loop
        while not all_ended and t < max_timesteps:
            if u is None:
                u_encoding = self.encoder(x)
            else:
                u_encoding = u[:, t, :]

            u_encoding = u_encoding.permute(1, 0, 2)  # (T, B, D)

            d_t, d_n = self.d_forward_recurrent(u_encoding, d_t)
            d_t = d_t.squeeze(0)

            if self.gated_stochastic_transfer:
                h_p = self.gru_cell(torch.cat([d_t, z_t], dim=-1), h_p)
            else:
                h_p = torch.cat([d_t, z_t], dim=-1)  # Elman RNN

            # prior conditioned on d_t and z_{t-1}
            prior_mu_t, prior_sd_t = self.prior(h_p)

            # sampling and reparameterization
            if use_mode:
                z_t = prior_mu_t
            else:
                z_t = rsample_gaussian(prior_mu_t, prior_sd_t)

            if self.use_phi_z:
                z_t = self.phi_z(z_t)

            dec = self.decoder(torch.cat([z_t, d_t], dim=-1))

            dec = self.dropout(dec) if self.dropout is not None else dec

            parameters = self.likelihood(dec)  # (B, T, D)

            x = self.likelihood.sample(parameters)  # (B, T, D)
            # x = x.permute(0, 2, 1)

            all_x.append(x)
            all_z.append(z_t)

            x = x.unsqueeze(1)
            d_t = d_t.unsqueeze(0)

            # Update sequence length
            x_sl += seq_active

            # Check for sequence ending
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

        x = torch.stack(all_x, dim=1)

        outputs = SimpleNamespace(h_p=h_p)
        return (x, x_sl), outputs


class SRNNAudio(BaseModel):
    def __init__(
        self,
        likelihood: Union[str, nn.Module],
        input_size: int = 200,
        hidden_size: int = 256,
        latent_size: int = 64,
        dropout: float = 0,
        residual_posterior: bool = False,
        smoothing: bool = True,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        """An SRNN for modelling audio waveforms."""
        super().__init__()
        self.likelihood = likelihood
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.dropout = dropout
        self.residual_posterior = residual_posterior
        self.num_mix = num_mix
        self.num_bins = num_bins
        self.smoothing = smoothing

        if isinstance(likelihood, str):
            if likelihood == "DMoL":
                likelihood_module = DiscretizedLogisticMixtureDense(
                    x_dim=3 * num_mix,
                    y_dim=1,
                    num_mix=10,
                    num_bins=2 ** 16,
                )
            elif likelihood == "GMM":
                likelihood_module = DiagonalGaussianMixtureDense(
                    x_dim=3 * num_mix,
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
            nn.Linear(2 * hidden_size + latent_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size * likelihood_module.out_features),  # (B, T/S, S*D)
            nn.LeakyReLU(),
            View(-1, likelihood_module.out_features),  # (B, T, D)
        )
        self.srnn = SRNN(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood_module,
            x_dim=hidden_size,
            h_dim=hidden_size,
            z_dim=latent_size,
            dropout=dropout,
            residual_posterior=residual_posterior,
            smoothing=smoothing,
        )

        self.forward_split = self.forward

    def split_sequence(
        self, x: TensorType["B", "T", "C"], x_sl: TensorType["B"], length: int, drop_inactive: bool = False
    ):
        """Split a long sequence into smaller subsequences for memory constrained `.forward()`.

        We ensure that each split is wholly strideable `(i-stack_frames) % stack_frames == 0` via `get_modulo_length`.
        We do not overlap splits since there is no overlap between in the observed variable for SRNN.
        """
        length = get_modulo_length(length, self.input_size, kernel_size=self.input_size)
        splits_x, splits_x_sl = split_sequence(x, x_sl, length=length, overlap=0, drop_inactive=drop_inactive)
        return splits_x, splits_x_sl

    def forward(
        self,
        x: TensorType["B", "T", "D", float],
        x_sl: TensorType["B", int],
        beta: float = 1,
        free_nats: float = 0,
        d_0: TensorType["num_layers", "B", "h_dim"] = None,
        a_0: TensorType["num_layers", "B", "h_dim"] = None,
        z_0: TensorType["B", "h_dim"] = None,
    ):
        loss, metrics, outputs = self.srnn(x=x, x_sl=x_sl, d_0=d_0, a_0=a_0, z_0=z_0, beta=beta, free_nats=free_nats)
        outputs.x_hat = self.srnn.likelihood.sample(outputs.parameters)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "T", "x_dim"] = None,
        u: TensorType["B", "T", "x_dim"] = None,
        d_0: TensorType["num_layers", "B", "h_dim"] = None,
        a_0: TensorType["num_layers", "B", "h_dim"] = None,
        z_0: TensorType["B", "z_dim"] = None,
    ):
        x = torch.zeros(n_samples, 1, self.input_size, device=self.device) if x is None else x
        return self.srnn.generate(
            x=x,
            u=u,
            d_0=d_0,
            a_0=a_0,
            z_0=z_0,
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode=use_mode,
        )
