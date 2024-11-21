import torch
from torch.nn import Module, RNN, GRU, LSTM, Dropout
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from typing import Literal


class RNNFamily(Module):
    def __init__(
        self,
        model_name: Literal["RNN", "GRU", "LSTM"],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        # decide in forward pass wheter to concatenate the first and the last hidden state
        self.bidirectional = bidirectional

        rnn_variants = {"RNN": RNN, "GRU": GRU, "LSTM": LSTM}
        self.model = rnn_variants[model_name](
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # introduces a Dropout layer on the outputs of each RNN layer except the last layer
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # dropout of the last output layer
        self.dropout = Dropout(dropout)

    def forward(self, X: PackedSequence):
        # the second output is the all hiden states across layers
        output, _ = self.model(X)

        # `lens_unpacked` is the list recording the original sequences' length before zero padding.
        # `seq_unpacked` is the zero padded list of sequences.
        seq_unpacked, lens_unpacked = pad_packed_sequence(output, batch_first=True)

        final_output = (
            torch.stack(
                [
                    # pick the first and the last token's hidden state as the final output and
                    # concatenate them
                    torch.cat([seq_unpacked[i, 0, :], seq_unpacked[i, seq_len - 1, :]])
                    for (i, seq_len) in enumerate(lens_unpacked)
                ]
            )
            if self.bidirectional
            # pick the last token's hidden state as the final output
            else torch.stack(
                [
                    seq_unpacked[i, seq_len - 1, :]
                    for (i, seq_len) in enumerate(lens_unpacked)
                ]
            )
        )

        return self.dropout(final_output)

    @property
    def output_size(self):
        return (
            # concat the first and the last hidden state, both's size are double as each of
            # which concate the output from two directions.
            self.hidden_size * 4
            if self.bidirectional
            # only use the last hidden state
            else self.hidden_size
        )


class MLP(Module):
    def __init__(self):
        pass
