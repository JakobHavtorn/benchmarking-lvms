import math
import warnings

from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from torchtyping import TensorType
from blvm.evaluation.metrics import (
    BitsPerDimMetric,
    KLMetric,
    LLMetric,
    LatestMeanMetric,
    LossMetric,
)

from blvm.models import BaseModel
from blvm.models.wavenet.wavenet_modules import CausalConv1d, ResidualStack
from blvm.modules.convenience import AddConstant
from blvm.modules.distributions import (
    ConditionalDistribution,
    DiagonalGaussianDense,
    DiagonalGaussianMixtureDense,
    DiscretizedLogisticMixtureDense,
)
from blvm.utils.operations import sequence_mask, stack_tensor, unstack_tensor
from blvm.utils.variational import discount_free_nats, kl_divergence_gaussian, kl_divergence_gaussian_mc, rsample_gaussian, precision_weighted_gaussian


class DiagonalGaussianDenseSTCN(DiagonalGaussianDense):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: nn.Module = nn.LeakyReLU,
        init_sd_mean: float = 1,
        epsilon: float = 1e-3,
    ) -> None:
        ConditionalDistribution.__init__(self)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.init_sd_mean = init_sd_mean
        self.epsilon = epsilon

        self.transform_mu = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.transform_sd = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, hidden_channels),
            activation(),
            nn.Linear(hidden_channels, out_channels),
        )

        self.sd_act = nn.Sequential(
            nn.Softplus(beta=math.log(2) / (init_sd_mean - epsilon)),
            AddConstant(epsilon),
        )

    def forward(self, x):
        mu = self.transform_mu(x)
        sd = self.transform_sd(x)
        sd = self.sd_act(sd)
        return mu, sd

class STCN(BaseModel):
    def __init__(
        self,
        likelihood: str = "DMoL",
        in_channels: int = 1,
        n_layers: int = 5,
        n_stacks: Optional[int] = None,
        latent_size: List[int] = [256, 128, 64, 32, 16],  # n_stacks
        res_channels: int = 256,
        kernel_size: int = 2,
        base_dilation: int = 2,
        n_stack_frames: int = 1,
        precision_posterior: bool = True,
        dense: bool = True,
        top_down: bool = True,
    ) -> None:
        """Stochastic autoregressive modelling of audio with latent variables and dilated convolutions [2].

        The total number of residual blocks (layers) used is equal to the n_layers times the n_stacks.
        This is `k` in Figure 4 in the paper [1].

        An illustration of the model:

                             |---------------------------------------------------| *residual*
                             |                                                   |
                             |             |-- tanh --|                          |
                 -> *input* ---> dilate ---|          * ---> 1x1 ---|----------- + ---> *input*
                                           |-- sigm --|             |
                                                                    |
                                                                    |
                 -> *skip* ---------------------------------------- + ----------------> *skip*

        # of parameters in original STCN-dense(GMM): 15.296.021
        # of parameters in 1-layered STCN with original code: 6.129.237.0

        Args:
            likelihood (nn.Module): Module for computing parameters of the output distribution and the loglikelihood.
            in_channels (int): Number of embeddings to optionally use for input. If 1, convolve input without embedding.
            n_layers (int): Number of residual blocks per stack. Dilations chosen as 2, 4, 8, 16, 32, 64...
            n_stacks (Optional[int]): Number of stacks of residual blocks. Defaults to number of latent variables.
            n_latents (int): Number of latent variables.
            res_channels (int): Number of channels in residual blocks (and embedding if num_embeddings > 1).
            dense (bool): If False, use only bottom latent variable to infer an observed timestep. Defaults to True.
            top_down (bool): If True, use top-down inference otherwise bottom-up. Defaults to True.

        Reference:
            [1] WaveNet: A Generative Model for Raw Audio (https://arxiv.org/abs/1609.03499)
            [2] STCN: Stochastic Temporal Convolutional Networks (https://arxiv.org/abs/1902.06568)
        """
        super().__init__()

        n_latents = len(latent_size)
        n_stacks = len(latent_size) if n_stacks is None else n_stacks

        self.likelihood = likelihood
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        self.n_latents = n_latents
        self.latent_size = latent_size
        self.in_channels = in_channels
        self.res_channels = res_channels
        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        self.n_stack_frames = n_stack_frames
        self.precision_posterior = precision_posterior
        self.dense = dense
        self.top_down = top_down

        self.causal = CausalConv1d(
            in_channels=in_channels * n_stack_frames,
            out_channels=res_channels,
            kernel_size=kernel_size,
            activation=None,
        )

        # we use only every `n_stacks`` skip connections so some are left unused..
        self.res_stack = ResidualStack(
            n_layers=n_layers,
            n_stacks=n_stacks,
            res_channels=res_channels,
            kernel_size=kernel_size,
            base_dilation=base_dilation,
        )

        self.receptive_fields = [rf + self.causal.kernel_size - 1 for rf in self.res_stack.receptive_fields]
        self.receptive_field = self.receptive_fields[-1]

        self.prior = [None] * self.n_latents
        self.posterior = [None] * self.n_latents
        order = reversed(range(self.n_latents)) if self.top_down else range(self.n_latents)
        for i, l in enumerate(order):
            if i == 0:
                in_channels = self.res_channels
            else:
                l_cond = l + 1 if self.top_down else l - 1
                in_channels = self.res_channels + self.latent_size[l_cond]
            self.prior[l] = DiagonalGaussianDenseSTCN(in_channels, self.latent_size[l], res_channels, init_sd_mean=0.5)
            self.posterior[l] = DiagonalGaussianDenseSTCN(in_channels, self.latent_size[l], res_channels, init_sd_mean=0.1)

        self.prior = nn.ModuleList(self.prior)
        self.posterior = nn.ModuleList(self.posterior)

        if dense:
            out_transform_in_size = sum(latent_size)
        else:
            out_transform_in_size = latent_size[0]

        self.out_transform = ResidualStack(
            n_layers=n_layers,
            n_stacks=1,
            res_channels=res_channels,
            in_channels=out_transform_in_size,
            kernel_size=kernel_size,
            base_dilation=1,  # no dilation in output blocks
        )
        self.inv_std = 1 / math.sqrt(self.n_stacks)

        if isinstance(likelihood, str):
            num_mix = 10
            if likelihood == "DMoL":
                likelihood_module = DiscretizedLogisticMixtureDense(
                    x_dim=2 * num_mix + num_mix,
                    y_dim=1,
                    num_mix=num_mix,
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
                likelihood_module = DiagonalGaussianDense(x_dim=2, y_dim=1, epsilon=1e-4)
            else:
                raise ValueError(f"Unknown likelihood type {likelihood}")

        self.out_upsample = nn.Sequential(
            nn.Linear(res_channels, likelihood_module.out_features * n_stack_frames),
            nn.ReLU(),
        )

        self.likelihood_module = likelihood_module

    def build_metrics(
        self,
        loss: TensorType[-1],
        elbo: TensorType["B"],
        log_prob: TensorType["B"],
        kld: TensorType["B"],
        klds: List[TensorType["B"]],
        mu_q: List[TensorType["B", "T", "C"]],
        x_sl: TensorType["B"],
        beta: float,
        free_nats: float,
    ):
        z_sl = torch.div(x_sl, self.n_stack_frames, rounding_mode="floor")

        klds_nats = [KLMetric(klds[l], name=f"kl_{l} (nats)", log_to_console=False) for l in range(self.n_latents)]
        klds_bpz = [KLMetric(klds[l] / math.log(2), name=f"kl_{l} (bpz)", reduce_by=z_sl) for l in range(self.n_latents)]
        klds_bpx = [KLMetric(klds[l] / math.log(2), name=f"kl_{l} (bpx)", reduce_by=x_sl) for l in range(self.n_latents)]

        metrics = [
            LossMetric(loss, weight_by=elbo.numel()),
            BitsPerDimMetric(elbo, name="elbo (bpx)", reduce_by=x_sl),
            LLMetric(elbo, name="elbo (nats)"),
            LatestMeanMetric(beta, name="beta"),
            LatestMeanMetric(free_nats, name="free_nats"),
            LLMetric(log_prob, name="rec (nats)", log_to_console=False),
            BitsPerDimMetric(log_prob, name="rec (bpx)", reduce_by=x_sl),
            KLMetric(kld, name="kl (nats)", log_to_console=False),
            KLMetric(kld / math.log(2), name="kl (bpz)", reduce_by=z_sl),
            *klds_nats,
            *klds_bpz,
            *klds_bpx,
        ]
        return metrics

    def compute_loss(
        self,
        y: TensorType["B", "T", "C"],
        x_sl: TensorType["B", int],
        parameters: Tuple[TensorType["B", "T", "C"]],
        mu_p: List[TensorType["B", "T", "C"]],
        sd_p: List[TensorType["B", "T", "C"]],
        mu_q: List[TensorType["B", "T", "C"]],
        sd_q: List[TensorType["B", "T", "C"]],
        z: List[TensorType["B", "T", "C"]],
        free_nats: float,
        beta: float,
    ):
        """Compute the loss as negative log-likelihood per frame, masked and mormalized according to sequence lengths.

        Args:
            y (torch.LongTensor): Input audio waveform, i.e. the y (B, T) of quantized integers.
            x_sl (torch.LongTensor): Sequence lengths of examples in the batch.
            parameters (torch.FloatTensor): Parameters for output distribution (likelihood).
        """
        # Observed variable
        log_prob_twise = self.likelihood_module.log_prob(y, parameters)  # (B, T, n_stack_frames)

        seq_mask = sequence_mask(x_sl, device=log_prob_twise.device)  # (B, T)
        log_prob = (log_prob_twise * seq_mask).sum(1)  # (B,)

        # Latent variables
        z_mask = seq_mask[:, ::self.n_stack_frames].unsqueeze(-1)
        if self.top_down:
            klds = [kl_divergence_gaussian(mu_q[l], sd_q[l], mu_p[l], sd_p[l]) * z_mask for l in range(self.n_latents)]
        else:
            klds = [kl_divergence_gaussian_mc(mu_q[l], sd_q[l], mu_p[l], sd_p[l], z[l]) * z_mask for l in range(self.n_latents)]
        klds_fn = [discount_free_nats(klds[l], free_nats, shared_dims=-1) * z_mask for l in range(self.n_latents)]
        kld = torch.cat(klds, dim=-1).sum((1, 2))  # (B, T)
        kld_fn = torch.cat(klds_fn, dim=-1).sum((1, 2))  # (B, T)
        klds = [kl.sum((1, 2)) for kl in klds]
        klds_fn = [kl.sum((1, 2)) for kl in klds_fn]

        elbo = log_prob - kld  # (B,)

        loss = -(log_prob - beta * kld_fn).sum() / x_sl.sum()  # (1,)
        return loss, elbo, log_prob, kld, klds

    def infer(self, d: TensorType["S", "B", "C", "T", float]):
        # select every self.n_latents skip connections
        d = d[self.n_latents - 1 :: self.n_latents]  # (S // self.n_latents, B, C, T + 1)
        # select prior and posterior inputs shifted by one frame
        d_p = [d_[..., :-1].permute(0, 2, 1) for d_ in d]  # (S // self.n_latents, B, T, C)
        d_q = [d_[..., 1:].permute(0, 2, 1) for d_ in d]  # (S // self.n_latents, B, T, C)

        mu_p, sd_p = [None] * self.n_latents, [None] * self.n_latents
        mu_q, sd_q = [None] * self.n_latents, [None] * self.n_latents
        z = [None] * self.n_latents  # (S // self.n_latents, B, T, C)

        order = reversed(range(self.n_latents)) if self.top_down else range(self.n_latents)
        for i, l in enumerate(order):
            if i == 0:
                in_p = d_p[l]
                in_q = d_q[l]
            else:
                l_cond = l + 1 if self.top_down else l - 1
                in_p = torch.cat([d_p[l], z[l_cond]], dim=-1)
                in_q = torch.cat([d_q[l], z[l_cond]], dim=-1)

            mu_p[l], sd_p[l] = self.prior[l](in_p)
            mu_q[l], sd_q[l] = self.posterior[l](in_q)
            if self.precision_posterior:
                mu_q[l], sd_q[l] = precision_weighted_gaussian(mu_p[l], sd_p[l], mu_q[l], sd_q[l])
            z[l] = rsample_gaussian(mu_q[l], sd_q[l])

        return mu_p, sd_p, mu_q, sd_q, z

    def split_sequence(self, x: TensorType["B", "T", "C", float], x_sl: TensorType["B", int], length: int):
        """Split a long sequence into smaller subsequences for memory constrained `.forward()`."""
        raise NotImplementedError()

    def forward_split(
        self,
        x: TensorType["B", "T", "C", float],
        x_sl: TensorType["B", int],
        i_split: int,
        y: TensorType["B", "T"] = None,
    ):
        """A convience forward method for split input sequences"""
        return self.forward(x, x_sl, y=y, pad_receptive_field=(i_split == 0))

    def forward(
        self,
        x: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        y: TensorType["B", "T"] = None,
        pad_receptive_field: bool = True,
        free_nats: float = 0,
        beta: float = 1,
    ):
        """Reconstruct an input and compute the log-likelihood loss.

        The duration of x has to be longer than the receptive field if `pad_receptive_field` is `False`.

        Args:
            x (torch.Tensor): Audio waveform (batch, timestep, channels) with values in [-1, 1]
            x_sl (torch.LongTensor): Sequence lengths of each example in the batch.
            y (torch.Tensor): An optional custom y. Must have shape matching x after convolving (consumes RF).
            pad_causal (bool): Shifts input one frame to the left compared to y by removing the right-most input.
            pad_receptive_field (bool): If True, pads receptive field to left of input to predict on all of input.
                                        If False, does not predict on first receptive field number of inputs.
            free_nats (float): Free nats for the latent variables. Defaults to 0.
            beta (float): KL-annealing constsant. Defaults to 1.
        """
        if y is None:
            y = x.detach()
            if not pad_receptive_field:
                # Remove receptive field from x (as y) if not padding since we condition on these values.
                y = y[:, self.receptive_field * self.n_stack_frames :]

        if self.n_stack_frames > 1:
            x, p = stack_tensor(x, self.n_stack_frames, dim=1)

        x = x.unsqueeze(-1) if x.ndim == 2 else x  # (B, T) -> (B, T, 1)
        y = y.unsqueeze(-1) if y.ndim == 2 else y

        x = x.transpose(1, 2)  # (B, C, T)
        if pad_receptive_field:
            T = x.size(2)
            x = nn.functional.pad(x, (self.receptive_field, 0))  # gives `rf - 1` extra outputs
            if T < self.receptive_field:
                warnings.warn(f"Padded input of shape {x.shape} with a larger receptive_field={self.receptive_field}.")
        else:
            T = x.size(2) - self.receptive_field
            x_sl = x_sl - self.n_stack_frames * self.receptive_field
            if x.size(2) <= self.receptive_field:
                raise ValueError(f"Input must be at least as long as the receptive field if {pad_receptive_field=}")

        output = self.causal(x, pad_causal=False)  # (B, C, T + rf - (causal.kernel_size - 1))
        skip_connections = self.res_stack(output, skip_size=T + 1)  # (S, B, C, T + rf - rf + 1)

        mu_p, sd_p, mu_q, sd_q, z = self.infer(skip_connections)  # (B, C, T)

        if self.dense:
            logits_in = torch.cat(z, dim=-1)
        else:
            logits_in = z[0]

        logits_in = logits_in.permute(0, 2, 1)  # (B, C, T)
        logits_in = nn.functional.pad(logits_in, (self.out_transform.receptive_field - 1, 0))  # (B, C, T + rf - 1)
        skip_logits = self.out_transform(logits_in, skip_size=T)  # (S, B, C, T)

        logits = sum(skip_logits) * self.inv_std  # (B, C, T)

        logits = self.out_upsample(logits.permute(0, 2, 1))
        if self.n_stack_frames > 1:
            logits = unstack_tensor(logits, self.n_stack_frames, p, dim=-1)  # (B, T, C)

        params = self.likelihood_module(logits)

        loss, elbo, log_prob, kld, klds = self.compute_loss(y, x_sl, params, mu_p, sd_p, mu_q, sd_q, z, free_nats, beta)

        metrics = self.build_metrics(loss, elbo, log_prob, kld, klds, mu_q, x_sl, beta, free_nats)

        reconstructions = self.likelihood_module.sample(params)
        reconstructions_mode = self.likelihood_module.mode(params)  # (B, T, C)

        z_sl = [torch.ceil(x_sl / self.n_stack_frames).long()] * self.n_stacks
        output = SimpleNamespace(
            loss=loss,
            elbo=elbo,
            klds=klds,
            log_prob=log_prob,
            z=z,
            z_sl=z_sl,
            enc_mus=mu_q,
            prior_mus=mu_p,
            params=params,
            y=y,
            reconstructions=reconstructions,
            reconstructions_mode=reconstructions_mode,
        )
        return loss, metrics, output

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode_observations: bool = False,
        x: Optional[TensorType["B", "T", "C"]] = None,
    ):
        raise NotImplementedError()
