import torch
import torch.nn as nn

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from blvm.modules.dropout import Dropout1dPackedData


class LSTMBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = True,
        sum_directions: bool = False,
        dropout_prob: float = 0.40,
        temporal_dropout: bool = True,
        return_all: bool = False,
    ):
        """
        Implements multiple consecutive LSTM layers.
        Args:
            input_size (int): Size of the input feature dimension.
            hidden_size (int): Size of the output feature dimension of each LSTM layer.
            num_layers (int): The number of LSTM layers.
            bidirectional (bool): Whether to use a bidirectional LSTM.
            sum_directions (bool): If True, will sum the output from the forward and backward direction.
            dropout_prob (float): The dropout rate applied to the output of each LSTM layer.
            temporal_dropout (bool): If True, uses variational dropout (i.e., 1D dropout).
            return_all (bool): If True, returns he outpur from all LSTM layers.
        """
        super(LSTMBlock, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.sum_directions = sum_directions
        self.p = dropout_prob
        self.temporal_dropout = temporal_dropout
        self.return_all = return_all

        if sum_directions and not bidirectional:
            raise ValueError("LSTM block must be bidirectional to sum directions.")

        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        bd_scale = 2 if bidirectional else 1
        sd_scale = 2 if sum_directions else 1

        for idx in range(num_layers):
            lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional)
            dropout_layer = Dropout1dPackedData(p=dropout_prob) if temporal_dropout else nn.Dropout(p=dropout_prob)
            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(dropout_layer)
            input_size = (hidden_size * bd_scale) // sd_scale

    def forward(self, input, seq_lens):
        """
        Args:
            input (Tensor): Input of shape (T, N, F) with dtype == float32.
            seq_lens (Tensor): The sequence lengths of the input of size (N) with dtype == int64.

        Returns:
            Tensor: Output of shape (T, N, F) with F = hidden_size or hidden_size x 2 if sum_directions is False.
        """
        x = input if isinstance(input, PackedSequence) else pack_padded_sequence(input, seq_lens)
        outputs = []
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm_layer(x)
            if self.sum_directions:
                sd_data = x.data.view(-1, 2, self.hidden_size).sum(dim=1)
                x = torch.nn.utils.rnn.PackedSequence(data=sd_data, batch_sizes=x.batch_sizes)
            do_data = dropout_layer(x.data)
            x = torch.nn.utils.rnn.PackedSequence(data=do_data, batch_sizes=x.batch_sizes)
            outputs.append(x)

        if self.return_all:
            return [pad_packed_sequence(x)[0] for x in outputs], seq_lens
        return pad_packed_sequence(x)  # pad_packed_sequences returns seq_lens
