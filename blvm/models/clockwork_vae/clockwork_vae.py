import math

from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit

from torchtyping import TensorType

from blvm.evaluation.metrics import (
    BitsPerDimMetric,
    EMAMetric,
    KLMetric,
    LLMetric,
    LatestMeanMetric,
    LossMetric,
)
from blvm.models.base_model import BaseModel
from blvm.models.clockwork_vae.convolutional_coders import ConvCoder1d
from blvm.modules.distributions import DiagonalGaussianDense, DiagonalGaussianMixtureDense, DiscretizedLogisticMixtureDense
from blvm.modules.rssm import RSSMCell
from blvm.utils.padding import get_modulo_length, get_same_padding, pad_modulo
from blvm.utils.variational import discount_free_nats, kl_divergence_gaussian
from blvm.utils.operations import sequence_mask, split_sequence


class CWVAE(nn.Module):
    def __init__(
        self,
        z_size: Union[int, List[int]],
        h_size: Union[int, List[int]],
        strides: List[int],
        encoder: nn.Module,
        decoder: nn.Module,
        likelihood: nn.Module,
        g_size: Optional[int] = 0,
        residual_posterior: bool = False,
        precision_posterior: bool = False,
        with_resets: bool = False,
        jit_compile: bool = True,
    ):
        """Clockwork VAE like latent variable model.

        All same padding is done per level in the encoder/decoder and is always put on the right side.

        Args:
            z_size (Union[int, List[int]]): Size(s) of the temporal latent variables.
            h_size (Union[int, List[int]]): Size(s) of the temporal deterministic variables.
            g_size (Optional[int]): Size of the optional global latent variable. Defaults to 0.
            strides (List[int]): Strides per layer.
            encoder (nn.Module): Transformation used to infer deterministic representations of input.
            decoder (nn.Module): Transformation used to decode context output by the final layer.
            likelihood (nn.Module): Transformation used to evaluate the likelihood (log_prob) of the reconstruction.
            residual_posterior (bool, optional): If True, compute mu_q = mu_q' + mu_p. Defaults to False.
            num_rssm_gru_cells (int, optional): Number of stacked GRU cells used per RSSM cell. Defaults to 1.
            with_resets (bool, optional): Reset state whenever layer above ticks (never reset top). Defaults to False.
        """
        super().__init__()

        assert isinstance(strides, list)

        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood
        self.residual_posterior = residual_posterior
        self.precision_posterior = precision_posterior
        self.g_size = g_size
        self.with_resets = with_resets
        self.jit_compile = jit_compile

        self.num_levels = len(strides)
        self.strides = strides
        self.overall_strides = np.cumprod(strides)
        self.overall_stride = self.overall_strides[-1]
        self.overall_receptive_field = self.encoder.overall_receptive_field
        self.overall_receptive_fields = self.encoder.overall_receptive_fields
        self.receptive_fields = self.encoder.receptive_fields

        self.e_size = self.encoder.e_size
        self.z_size = [z_size] * self.num_levels if isinstance(z_size, int) else z_size
        self.h_size = [h_size] * self.num_levels if isinstance(h_size, int) else h_size
        self.c_size = [e_size for e_size in self.decoder.e_size[1:]] + [0]

        assert (len(self.z_size) == len(self.h_size) == len(self.c_size)), f"{self.z_size=}=={self.h_size=}=={self.c_size=}"

        cells = []
        for h_dim, z_dim, c_dim, e_dim in zip(self.h_size, self.z_size, self.c_size, self.e_size):
            cell = RSSMCell(
                    h_dim=h_dim,
                    z_dim=z_dim,
                    c_dim=c_dim,
                    e_dim=e_dim,
                    residual_posterior=residual_posterior,
                    precision_posterior=precision_posterior,
                )
            if jit_compile:
                cell = jit.script(cell)
            cells.append(cell)

        self.cells = nn.ModuleList(cells)

    def build_metrics(self, loss, elbo, log_prob, kld, kld_l, x_sl, beta, free_nats):
        kld_layers_metrics_nats = [
            KLMetric(kld_l[l], name=f"kl_{l} (nats)", log_to_console=False) for l in range(self.num_levels)
        ]
        kld_layers_metrics_bpd = [
            KLMetric(kld_l[l] / math.log(2), name=f"kl_{l} (bpt)", reduce_by=(x_sl / self.overall_strides[l]))
            for l in range(self.num_levels)
        ]

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            EMAMetric(-elbo / math.log(2), name="elbo ema (bpt)", reduce_by=x_sl, weight_by=0.97),
            LLMetric(elbo, name="elbo (nats)"),
            BitsPerDimMetric(elbo, name="elbo (bpt)", reduce_by=x_sl),
            LLMetric(log_prob, name="rec (nats)", log_to_console=False),
            BitsPerDimMetric(log_prob, name="rec (bpt)", reduce_by=x_sl),
            KLMetric(kld, name="kl (nats)", log_to_console=False),
            KLMetric(kld / math.log(2), name="kl (bpt)", reduce_by=x_sl / self.overall_strides[0]),
            *kld_layers_metrics_nats,
            *kld_layers_metrics_bpd,
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
        ]

        return metrics

    def compute_elbo(
        self,
        y: TensorType["B", "T", "D"],
        seq_mask: TensorType["B", "T", int],
        level_masks: List[TensorType["B", "T", int]],
        x_sl: TensorType["B", int],
        parameters: TensorType["B", "T", "D"],
        kld_layerwise: List[TensorType["B", "T", "latent_size"]],
        beta: float = 1,
        free_nats: float = 0,
    ):
        """Return reduced loss for batch and non-reduced ELBO, log p(x|z) and KL-divergence"""
        log_prob_twise = self.likelihood.log_prob(y, parameters) * seq_mask
        log_prob = log_prob_twise.view(y.size(0), -1).sum(1)  # (B,)

        # sum over time (masked) and latent dim
        kld_l, klds_fn = [], []
        for l in range(self.num_levels):
            mask = level_masks[l].unsqueeze(-1)
            fn = free_nats * self.overall_strides[l] / self.overall_strides[0]  # scale up relative to bottom z
            kld_l.append((kld_layerwise[l] * mask).sum((1, 2)))  # (B,)
            klds_fn.append((discount_free_nats(kld_layerwise[l], fn, shared_dims=-1) * mask).sum((1, 2)))  # (B,)

        kld, kld_fn = sum(kld_l), sum(klds_fn)  # sum over levels

        elbo = log_prob - kld  # (B,)

        loss = -(log_prob - beta * kld_fn).sum() / x_sl.sum()  # (1,)

        return loss, elbo, log_prob, kld, kld_l

    def split_sequence(
        self, x: TensorType["B", "T", "C", float], x_sl: TensorType["B", int], length: int, drop_inactive: bool = False
    ):
        """Split a long sequence into smaller subsequences for memory constrained `.forward()`.

        We ensure that each split is wholly strideable `(i-k) % s == 0` via `get_modulo_length`.
        We overlap splits with `rf - s` since this is the amount of overlap that would have been in a non-split conv.
        """
        length = get_modulo_length(length, self.overall_stride, self.overall_receptive_field)
        overlap = self.overall_receptive_field - self.overall_stride
        splits_x, splits_x_sl = split_sequence(x, x_sl, length=length, overlap=overlap, drop_inactive=drop_inactive)
        return splits_x, splits_x_sl

    def forward_split(
        self,
        x: TensorType["B", "T", "C", float],
        x_sl: TensorType["B", int],
        is_last_split: bool,
        state0: List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]] = None,
        beta: float = 1,
        free_nats: float = 0,
        y: Optional[Union[TensorType["B", "T", "D"], TensorType["B", "T"]]] = None,
        use_mode_global: bool = False,
    ):
        """A convience forward method for split input sequences"""
        return self.forward(
            x,
            x_sl,
            state0=state0,
            beta=beta,
            free_nats=free_nats,
            y=y,
            use_mode_global=use_mode_global,
            pad_strideable=False,
            pad_same=is_last_split,
        )

    def forward(
        self,
        x: Union[TensorType["B", "T", "D"], TensorType["B", "T"]],
        x_sl: TensorType["B", int],
        state0: List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]] = None,
        beta: float = 1,
        free_nats: float = 0,
        y: Optional[Union[TensorType["B", "T", "D"], TensorType["B", "T"]]] = None,
        pad_strideable: bool = False,
        pad_same: bool = True,
        use_mode_global: bool = False,
    ):
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if pad_strideable and not pad_same:
            # pad x to be fully strideable (only if not same padding)
            x = pad_modulo(x, self.overall_stride, self.overall_receptive_field, dim=1)

        if y is None:
            y = x.clone().detach()

        # Don't compute reconstruction loss on the part of x with no dedicated latent states
        x_sl = x_sl.cpu()  # no-op if already on CPU
        if not pad_same:
            # The amount of input consumed by convolutions
            same_padding = get_same_padding(x.shape[1], self.overall_receptive_field, self.overall_stride)
            x_sl = (x_sl - same_padding).clamp(0)
            y = y[:, :-same_padding]

        # Mask for x (reconstruction)
        seq_mask = sequence_mask(x_sl, dtype=bool, device=x.device)

        # Masks for z
        level_sl = []
        level_masks = []
        for l in range(self.num_levels):
            sl = (x_sl / self.overall_strides[l]).ceil().to(int)
            mask = sequence_mask((x_sl / self.overall_strides[l]).ceil().to(int), dtype=bool, device=x.device)
            level_sl.append(sl)
            level_masks.append(mask)

        # compute lengths of representations at the different levels if same padding and if not
        same_paddings = []
        for l in range(self.num_levels):
            input_length = math.ceil(x.shape[1] / self.strides[l-1]) if l > 0 else x.shape[1]
            padding = get_same_padding(input_length, kernel_size=self.receptive_fields[l], stride=self.strides[l])
            same_paddings.append(padding)

        # compute encodings
        encoder_right_pad = same_paddings if pad_same else [0] * self.num_levels
        encodings_list = self.encoder(x.permute(0, 2, 1), pad_right=encoder_right_pad)
        encodings = [enc.unbind(2) for enc in encodings_list]  # unbind time dimension (List[Tuple[Tensor]])

        # initial context for top layer
        context_l = [None] * len(encodings[-1])
        context_l = [self.cells[-1].get_empty_context(x.size(0))] * len(encodings[-1])

        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=x.size(0)) for cell in self.cells] if state0 is None else state0

        kld_l = [[] for _ in range(self.num_levels)]
        latents = [[] for _ in range(self.num_levels)]
        enc_mus = [[] for _ in range(self.num_levels)]
        prior_mus = [[] for _ in range(self.num_levels)]
        next_forward_state0 = [[] for _ in range(self.num_levels)]
        for l in range(self.num_levels - 1, -1, -1):
            if l < self.num_levels - 1:
                context_l = context_l.unbind(2)

            all_states = []
            all_distributions = []
            T_l = len(encodings[l]) if pad_same else len(context_l)  # if not padding same, ignore any consumed z
            for t in range(T_l):
                # reset stochastic state whenever the layer above ticks (never reset top)
                if self.with_resets and (l < self.num_levels - 1) and (t % self.strides[l + 1] == 0):
                    states[l] = self.cells[l].get_initial_state(batch_size=x.size(0))

                # cell forward
                states[l], distributions = self.cells[l](encodings[l][t], states[l], context_l[t])

                all_states.append(states[l])
                all_distributions.append(distributions)

            # store correct state for next forward call (i.e. according to approx. seq len in the respective layer)
            # all_states:             (T, S, B, D)  (S for state variable, z or h index)
            # next_forward_state0[l]:    (B, S, D)
            # next_forward_state0:    (L, S, B, D)
            example_stop_idx = ((x_sl / self.overall_strides[l]).ceil().to(int) - 1).clamp(0)  # minus one to get index
            states_z_final_t = torch.stack([all_states[t][0][b] for b, t in enumerate(example_stop_idx)])  # (B, S, D)
            states_h_final_t = torch.stack([all_states[t][1][b] for b, t in enumerate(example_stop_idx)])  # (B, S, D)
            next_forward_state0[l] = (states_z_final_t, states_h_final_t)

            # update context_l for below layer as cat(z_l, h_l)
            context_l = [torch.cat(all_states[t], dim=-1) for t in range(T_l)]

            # use context decoder to increase temporal resolution
            context_l = torch.stack(context_l, dim=2)  # (B, D, T)
            _, context_l = self.decoder[l](context_l, pad_right=same_paddings[l])

            # compute kl divergence
            enc_mu = torch.stack([all_distributions[t].enc_mu for t in range(T_l)], dim=1)
            enc_sd = torch.stack([all_distributions[t].enc_sd for t in range(T_l)], dim=1)
            prior_mu = torch.stack([all_distributions[t].prior_mu for t in range(T_l)], dim=1)
            prior_sd = torch.stack([all_distributions[t].prior_sd for t in range(T_l)], dim=1)

            latents[l] = torch.stack([all_distributions[t].z for t in range(T_l)], dim=1)
            enc_mus[l] = enc_mu
            prior_mus[l] = prior_mu

            kld_l[l] = kl_divergence_gaussian(enc_mu, enc_sd, prior_mu, prior_sd)

        dec = context_l.permute(0, 2, 1)
        parameters = self.likelihood(dec)
        reconstruction = self.likelihood.sample(parameters)
        reconstruction_mode = self.likelihood.mode(parameters)

        loss, elbo, log_prob, kld, kld_l = self.compute_elbo(
            y, seq_mask, level_masks, x_sl, parameters, kld_l, beta, free_nats
        )

        metrics = self.build_metrics(loss, elbo, log_prob, kld, kld_l, x_sl, beta, free_nats)

        outputs = SimpleNamespace(
            elbo=elbo,
            log_prob=log_prob,
            kld=kld,
            y=y,
            seq_mask=seq_mask,
            z=latents,
            z_sl=level_sl,
            enc_mus=enc_mus,
            prior_mus=prior_mus,
            reconstructions=reconstruction,
            reconstructions_mode=reconstruction_mode,
            reconstructions_parameters=parameters,
            state_n=next_forward_state0,
        )

        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode_observations: bool = False,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        # initial RSSM state (z, h)
        states = [cell.get_initial_state(batch_size=n_samples) for cell in self.cells] if state0 is None else state0

        # initial context_l for top layer
        context_l = [None] * (max_timesteps // self.overall_strides[-1])

        # compute lengths of representations at the different levels if same padding and if not
        same_paddings = []
        for l in range(self.num_levels):
            input_length = math.ceil(max_timesteps / self.strides[l-1]) if l > 0 else max_timesteps
            padding = get_same_padding(input_length, self.receptive_fields[l], self.strides[l])
            same_paddings.append(padding)

        for l in range(self.num_levels - 1, -1, -1):

            if l < self.num_levels - 1:
                context_l = context_l.unbind(2)

            all_states = []
            all_distributions = []
            T_l = max_timesteps // self.overall_strides[l] if l == self.num_levels - 1 else len(context_l)
            for t in range(T_l):
                # reset stochastic state whenever the layer above ticks (never reset top)
                if self.with_resets and (l < self.num_levels - 1) and (t % self.strides[l + 1] == 0):
                    states[l] = self.cells[l].get_initial_state(batch_size=x.size(0))

                # cell forward
                states[l], distributions = self.cells[l].generate(states[l], context_l[t])

                all_states.append(states[l])
                all_distributions.append(distributions)

            # update context_l for below layer as cat(z_l, h_l)
            context_l = [torch.cat(all_states[t], dim=-1) for t in range(T_l)]

            # use context decoder to increase temporal resolution
            context_l = torch.stack(context_l, dim=2)
            _, context_l = self.decoder[l](context_l, pad_right=same_paddings[l])

        dec = context_l.permute(0, 2, 1)
        parameters = self.likelihood(dec)
        x_sample = self.likelihood.sample(parameters)
        x_mode = self.likelihood.mode(parameters)
        x = x_mode if use_mode_observations else x_sample
        x_sl = torch.ones(x.size(0), dtype=torch.int) * max_timesteps
        outputs = SimpleNamespace()
        return (x, x_sl), outputs


class CWVAEAudio(BaseModel):
    def __init__(
        self,
        z_size: Union[int, List[int]] = 64,
        h_size: Union[int, List[int]] = 128,
        g_size: Optional[int] = 0,
        strides: Union[int, List[int]] = [64, 16, 16],
        dilations: Union[int, List[int]] = 1,
        residual_posterior: bool = False,
        precision_posterior: bool = False,
        num_level_layers: int = 3,
        stride_per_layer: int = 4,
        likelihood: str = "dmol",
        num_mix: int = 10,
        num_bins: int = 256,
        # norm_type: str = "ChannelwiseLayerNorm",
    ):
        super().__init__()

        self.z_size = z_size
        self.h_size = h_size
        self.g_size = g_size
        self.strides = strides
        self.dilations = dilations
        self.residual_posterior = residual_posterior
        self.precision_posterior = precision_posterior
        self.num_level_layers = num_level_layers
        self.stride_per_layer = stride_per_layer
        self.num_mix = num_mix
        self.num_bins = num_bins
        # self.norm_type = norm_type

        self.num_levels = len(strides)

        z_size = [z_size] * self.num_levels if isinstance(z_size, int) else z_size
        h_size = [h_size] * self.num_levels if isinstance(h_size, int) else h_size
        c_size = [h + z + g_size for h, z in zip(h_size, z_size)]

        assert all(h_size[0] == hs for hs in h_size)
        h_size = h_size[0]

        if isinstance(likelihood, str):
            if likelihood == "DMoL":
                likelihood = DiscretizedLogisticMixtureDense(
                    x_dim=h_size,  # 30
                    y_dim=1,
                    num_mix=num_mix,
                    num_bins=num_bins,
                )
            elif likelihood == "Gaussian":
                likelihood = DiagonalGaussianDense(
                    x_dim=h_size,
                    y_dim=1,
                    epsilon=1e-2,
                )
            elif likelihood == "GMM":
                likelihood = DiagonalGaussianMixtureDense(
                    x_dim=h_size,
                    y_dim=1,
                    num_mix=num_mix,
                    initial_sd=1,
                    epsilon=1e-2
                )
            else:
                raise ValueError(f"Unknown likelihood type {likelihood}")
        self.likelihood = likelihood

        encoder = ConvCoder1d(
            strides=strides,
            channels_in=1,
            channels=h_size,
            kernel_size=5,
            num_blocks=num_level_layers,
            # norm_type=norm_type,
            stride_per_block=stride_per_layer,
            transposed=False,
            block_type="BlockSeparable",
            activation=nn.ReLU
        )

        channels_out = [h_size] + [None] * (self.num_levels - 1)
        decoder = ConvCoder1d(
            strides=strides,
            channels_in=c_size,
            channels=h_size,
            channels_out=channels_out,
            kernel_size=5,
            num_blocks=num_level_layers,
            # norm_type=norm_type,
            stride_per_block=stride_per_layer,
            transposed=True,
            block_type="BlockSeparable",
            activation=nn.ReLU
        )

        self.cwvae = CWVAE(
            encoder=encoder,
            decoder=decoder,
            likelihood=likelihood,
            z_size=z_size,
            h_size=h_size,
            strides=strides,
            residual_posterior=residual_posterior,
            precision_posterior=precision_posterior,
            g_size=g_size,
        )
        self.overall_receptive_field = self.cwvae.overall_receptive_field
        self.overall_stride = self.cwvae.overall_stride
        self.split_sequence = self.cwvae.split_sequence
        self.forward_split = self.cwvae.forward_split

        # TODO
        # self.forward = self.cwvae.forward
        # self.generate = self.cwvae.generate

    def forward(
        self,
        x: TensorType["B", "T"],
        x_sl: TensorType["B", int],
        state0: List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]] = None,
        beta: float = 1,
        free_nats: float = 0,
        pad_strideable: bool = True,
        pad_same: bool = True,
        y: TensorType["B", "T"] = None,
    ):
        loss, metrics, outputs = self.cwvae(x, x_sl, state0, beta, free_nats, y, pad_strideable, pad_same)
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode_observations: bool = False,
        state0: Optional[List[Tuple[TensorType["B", "h_size"], TensorType["B", "z_size"]]]] = None,
    ):
        return self.cwvae.generate(
            n_samples=n_samples,
            max_timesteps=max_timesteps,
            use_mode_observations=use_mode_observations,
            state0=state0,
        )
