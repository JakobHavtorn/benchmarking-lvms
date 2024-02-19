from types import SimpleNamespace

import torch.nn as nn

from blvm.models.base_model import BaseModel
from blvm.evaluation import LossMetric, ErrorRateMetric
from blvm.modules.lstm_block import LSTMBlock
from blvm.utils.decoding import greedy_ctc
from blvm.data.tokenizers import word_tokenizer, char_tokenizer
from blvm.data.token_map import TokenMap
from blvm.data.tokens import BLANK_TOKEN


class SimpleLSTMASR(BaseModel):
    """A simple ASR model based on LSTMBlock and CTCLoss."""
    def __init__(
        self,
        token_map: TokenMap,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        sum_directions: bool = False,
        dropout_prob: float = 0.0,
        temporal_dropout: bool = True,
    ):
        super().__init__()
        self.output_size = len(token_map)
        self.token_map = token_map
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sum_directions = sum_directions
        self.dropout_prob = dropout_prob
        self.temporal_dropout = temporal_dropout

        self.blank_index = token_map.token2index[BLANK_TOKEN]

        self.lstm = LSTMBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            sum_directions=sum_directions,
            dropout_prob=dropout_prob,
            temporal_dropout=temporal_dropout,
        )
        self.output = nn.Linear(hidden_size * (bidirectional + 1), self.output_size)
        self.ctc_loss = nn.CTCLoss(blank=self.blank_index, reduction="none")

    def forward(self, x, x_sl, y, y_sl):
        """
        Args:
            input (Tensor): Input of size (B, H, T) with dtype == float32. Default is H = 80.
            seq_lens (Tensor): The sequence lengths of the input of size B with dtype == int64.

        Returns:
            Tensor:  Output of shape (T, B, F) with F = output_size.
            Tensor: 'seq_lens' reduced according to temporal stride.
        """
        x = x.permute(2, 0, 1)  # (B, I, T) to (T, B, I)
        z, z_sl = self.lstm(x, x_sl)
        return self.ctc_decoder(z, z_sl, y, y_sl)

    def ctc_decoder(self, z, z_sl, y, y_sl):
        logits = self.output(z)  # (T, B, O)
        log_probs = logits.log_softmax(dim=2)
        loss = self.ctc_loss(log_probs, y, z_sl, y_sl).sum() / y_sl.sum()

        hyps_raw = greedy_ctc(logits, z_sl, blank=self.blank_index)
        hyps_sl = [len(h) for h in hyps_raw]
        hyps = self.token_map.decode_batch(hyps_raw, hyps_sl, " ")
        refs = self.token_map.decode_batch(y, y_sl, " ")

        outputs = SimpleNamespace(logits=logits.transpose(0, 1), sl=z_sl, hyps=hyps, refs=refs)

        metrics = [
            LossMetric(loss, weight_by=y_sl.sum()),
            ErrorRateMetric(refs, hyps, word_tokenizer, name="wer"),
            ErrorRateMetric(refs, hyps, char_tokenizer, name="cer"),
        ]

        return loss, metrics, outputs
