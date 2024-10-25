####################################################################################################
#
####################################################################################################


import lightning as L
from omegaconf import DictConfig
import torch
from torch.nn import Module, Linear, RNN, GRU, LSTM
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import torch.nn.functional as F
from typing import Literal


class RNNFamily(Module):
    def __init__(self, 
                 model_name: Literal["RNN", "GRU", "LSTM"],
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 bidirectional: bool = False
                 ):
        super().__init__()

        if model_name == "RNN":
            self.model = RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif model_name == "GRU":
            self.model = GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            self.model = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional
            )

        self.model = self.model.to('cuda:0')


    def forward(self, X: PackedSequence):
        # the second output is the all hiden states across layers
        output, _ = self.model(X)

        # `lens_unpacked` is the list recording the original sequences' length before zero padding.
        # `seq_unpacked` is the zero padded list of sequences.
        seq_unpacked, lens_unpacked = pad_packed_sequence(output, batch_first=True)

        # pick the last token's hidden state as the final output
        final_output = torch.stack([
            seq_unpacked[i, seq_len-1, :]
            for (i, seq_len) in enumerate(lens_unpacked)
        ])

        return final_output


class HawkishDovishClassifier(L.LightningModule):
    def __init__(self, hparam: DictConfig, model):
        """
        The type of `model` is the subclass of `nn.Module`, i.e., nn.RNN.
        """
        super().__init__()
        self.hparam = hparam
        self.model = model

        # 3 is the number of classes
        if type(model) == RNNFamily:
            hidden_size = (
                hparam.RNNFamily.hidden_size * 2
                if hparam.RNNFamily.bidirectional
                else
                hparam.RNNFamily.hidden_size
            )
        else:
            hidden_size = hparam.MLP.output_size
    
        self.linear = Linear(hidden_size, 3).to(hparam.device)


    def forward(self, X):
        output = self.model(X)
        logits = self.linear(output)

        return logits


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparam.lr)


    def training_step(self, batch, batch_idx):
        X, y = batch['embed'], batch['label']
        loss = F.cross_entropy(self(X), y)

        self.log(
            "train_loss", 
            loss, 
            prog_bar = True, 
            on_epoch = True, 
            batch_size = self.hparam.classifier.batch_size
        )

        return loss


    # 1 validation step = 1 epoch
    def validation_step(self, batch, batch_idx):
        X, y = batch['embed'], batch['label']
        loss = F.cross_entropy(self(X), y)

        self.log(
            "val_loss", 
            loss, 
            prog_bar = True
        )

        return loss


    # 1 test step = 1 epoch
    def test_step(self, batch, batch_idx):
        X, y = batch['embed'], batch['label']
        logits = self(X)
        loss = F.cross_entropy(self(X), y)

        self.log(
            "test_loss", 
            loss, 
            prog_bar = True
        )

        # probs = F.softmax(logits, dim=1)
        # preds = torch.argmax(probs, axis=-1)

        return loss