from types import SimpleNamespace
from typing import Tuple, List

import torch
import torch.nn as nn

from torchtyping import TensorType

from blvm.modules.distributions import DiscretizedLogisticMixtureDense
from blvm.evaluation import LLMetric, BitsPerDimMetric
from blvm.evaluation.metrics import LossMetric
from blvm.utils.operations import sequence_mask, stack_tensor

from .base_model import BaseModel


class LSTMAudio(BaseModel):
    def __init__(
        self,
        stack_size: int = 64,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0,
        batch_first: bool = True,
        num_mix: int = 10,
        num_bins: int = 256,
    ):
        """A VRNN for modelling audio waveforms."""
        super().__init__()
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_mix = num_mix
        self.num_bins = num_bins

        self.embedding = nn.Sequential(
            nn.Linear(stack_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=True,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=False,
            proj_size=0,
        )
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3 * num_mix * stack_size),
            nn.ReLU(),
        )
        self.likelihood = DiscretizedLogisticMixtureDense(
            x_dim=3 * num_mix,
            y_dim=1,
            num_mix=num_mix,
            num_bins=num_bins,
        )

    def forward(
        self,
        x: TensorType["B", "T", float],
        x_sl: TensorType["B", int],
        s_0: Tuple[TensorType["L", "B", "H"], TensorType["L", "B", "H"]] = None,
    ):
        """Stacks the input waveform into vectors and predicts the next stack. Computes the loss on flat waveform."""
        x_sl = x_sl.cpu()
        x_sl_stack = (x_sl / self.stack_size).ceil().int()

        if s_0 is None:
            h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
            c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
            s_0 = (h_0, c_0)

        # stack waveform into vectors of `stack_size` frames
        x_stack, _ = stack_tensor(x, self.stack_size, dim=-1)

        # input and target include padding from stacking but this is handled by x_sl masking in log_prob
        x_input = x_stack[:, :-1]
        x_target = x_stack[:, 1:].detach().view(x.size(0), -1)

        e_stack = self.embedding(x_input)

        ps = torch.nn.utils.rnn.pack_padded_sequence(e_stack, x_sl_stack - 1, batch_first=self.batch_first)
        h, s_n = self.lstm(ps, s_0)
        h, h_sl = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=self.batch_first)

        self.dropout(h) if self.dropout else h

        o = self.decoder(h)

        # flatten to full waveform
        o = o.view(o.size(0), o.size(1) * self.stack_size, 3 * self.num_mix)

        parameters = self.likelihood(o)
        mode = self.likelihood.mode(parameters)
        sample = self.likelihood.sample(parameters)

        seq_mask = sequence_mask(x_sl, max_len=x_target.shape[1], device=x.device)
        log_prob = self.likelihood.log_prob(x_target.unsqueeze(-1), parameters)
        log_prob = (log_prob * seq_mask).sum(1)

        loss = -log_prob.sum() / x_sl.sum()

        metrics = [
            LossMetric(loss, weight_by=log_prob.numel()),
            LLMetric(log_prob),
            BitsPerDimMetric(log_prob, reduce_by=x_sl),
        ]
        outputs = SimpleNamespace(
            loss=loss,
            ll=log_prob,
            z=h,
            z_sl=x_sl_stack,
            reconstruction_sample=sample,
            reconstruction_mode=mode,
            s_n=s_n,
        )
        return loss, metrics, outputs

    def generate(
        self,
        n_samples: int = 1,
        max_timesteps: int = 100,
        use_mode: bool = False,
        x: TensorType["B", "x_dim"] = None,
        h0: TensorType["B", "h_dim"] = None,
    ):
        raise NotImplementedError()
