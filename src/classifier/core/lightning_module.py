####################################################################################################
# This file defines how the traing process should be proceed, such as what training loos is
# considered, what metris are logged during validation and test stages, and what optimizers should
# we adopt. `L.LightningModule` chunks the process into different pieces of events to allow easy
# configurations separately.
####################################################################################################


from functools import cached_property
import lightning as L
import torch
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    Tanh,
    ReLU,
    Sequential,
    Softmax,
)
import torch.nn.functional as F
from torchmetrics.classification import (
    PrecisionRecallCurve,
    AveragePrecision,
    Precision,
    Recall,
    F1Score,
    AUROC,
)
from typing import Union, Literal

from .metrics import ClassificationMetricsLogger
from .nn import RNNFamily, MLP


def get_nn(model_name: str, input_size, **nn_hparam) -> Union[RNNFamily, MLP]:
    if model_name == "mlp":
        return MLP()
    else:
        return RNNFamily(
            model_name,
            input_size=input_size,
            hidden_size=nn_hparam["hidden_size"],
            num_layers=nn_hparam["num_layers"],
            dropout=nn_hparam["dropout"],
            bidirectional=nn_hparam["bidirectional"],
        )


class HawkishDovishClassifier(L.LightningModule):
    num_classes = 3

    def __init__(
        self,
        pooling_strategy: Literal[
            "rnn", "sbert", "cls_non_trainable", "cls_trainable", "last_layer_mean"
        ],
        model_name: str,
        lr: float,
        class_weights: torch.Tensor,
        input_size: int,
        **nn_hparam,
    ):
        """
        The type of `model` is the subclass of `nn.Module`, i.e., nn.RNN.
        """
        super().__init__()

        self.input_size = input_size
        self.lr = lr
        self.class_weights = class_weights
        self.nn_hparam = nn_hparam

        if pooling_strategy == "rnn":
            self.nn = get_nn(model_name, input_size, **nn_hparam)
            self.nn_output_size = self.nn.output_size

            # classification layers (ff-> linear)
            self.ff = Sequential(
                Linear(self.nn_output_size, self.nn_output_size),
                Tanh(),
                Linear(self.nn_output_size, self.nn_output_size),
                Dropout(nn_hparam["ff_dropout"]),
            )
            self.linear = Linear(self.nn_output_size, self.num_classes)

            self.layernorm = LayerNorm(self.nn_output_size)

        else:
            if pooling_strategy == "sbert":
                self.proj = Sequential(
                    Linear(input_size, nn_hparam["ff_input_size"]),
                    ReLU(),
                    Dropout(nn_hparam["ff_dropout"]),
                )

                self.ff = Sequential(
                    Linear(nn_hparam["ff_input_size"], nn_hparam["ff_input_size"]),
                    ReLU(),
                    # Linear(nn_hparam["ff_input_size"], nn_hparam["ff_input_size"]),
                    Dropout(nn_hparam["ff_dropout"]),
                )

                self.linear = Linear(nn_hparam["ff_input_size"], self.num_classes)
            else:
                if pooling_strategy in ["cls_pooler", "last_layer_mean_pooler"]:
                    self.ff = Sequential(
                        Linear(input_size, input_size),
                        Tanh(),
                        Linear(input_size, input_size),
                        Dropout(nn_hparam["ff_dropout"]),
                    )
                self.linear = Linear(input_size, self.num_classes)

        self.softmax = Softmax(dim=-1)

        # validation and test metrics
        self.pr_curve = PrecisionRecallCurve("multiclass", num_classes=self.num_classes)
        self.macro_ap = AveragePrecision(
            "multiclass", num_classes=self.num_classes, average="macro"
        )
        self.ap = AveragePrecision(
            "multiclass", num_classes=self.num_classes, average=None
        )
        self.prec = Precision("multiclass", num_classes=self.num_classes, average=None)
        self.recall = Recall("multiclass", num_classes=self.num_classes, average=None)

        self.macro_f1 = F1Score(
            "multiclass", num_classes=self.num_classes, average="macro"
        )
        self.f1 = F1Score("multiclass", num_classes=self.num_classes, average=None)

        self.macro_auroc = AUROC(
            "multiclass", num_classes=self.num_classes, average="macro"
        )
        self.auroc = AUROC("multiclass", num_classes=self.num_classes, average=None)

    @cached_property
    def classes(self):
        return ["dovish", "hawkish", "neutral"]

    @cached_property
    def metrics_logger(self):
        return ClassificationMetricsLogger(self.logger)

    def forward(self, X):
        # pooling strategy: rnn
        # outputs = self.nn(X)
        # outputs = self.ff(self.layernorm(outputs))
        # logits = self.linear(outputs)

        # pooling strategy: cls_pooler or last_layer_mean_pooler
        # outputs = self.ff()
        # logits = self.linear(outputs)

        outputs = self.proj(X)
        logits = self.linear(outputs)

        # pooling strategy: cls or cls_pooler_output or last_layer_mean
        # logits = self.linear(X)

        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

    def on_train_start(self):
        self.logger.log_hyperparams(
            {"class_weights": self.class_weights, "learning_rate": self.lr}
            | self.nn_hparam
        )

    def training_step(self, batch, batch_idx):
        X, y = batch["embed"], batch["label"]

        # cross entropy loss
        # loss = F.cross_entropy(self(X), y, weight=self.class_weights.cuda())

        # hierarchical loss:
        logits = self(X)
        binary_logits = logits[:, 2]  # [batch_size, 1] for class3
        subclass_logits = logits[:, :2]  # [batch_size, 2] for class 1&2

        loss = 0
        # first stage (identify whether class 2 (neutral) or not)
        mask = y == 2
        if mask.any():
            loss += F.binary_cross_entropy_with_logits(
                binary_logits,
                mask.float(),
            )

        # second stage (given not class 3, identifying class 1 or 2)
        mask = y != 2
        if mask.any():
            subclass_binary_labels = (y[mask] == 0).float()

            subclass_loss = F.binary_cross_entropy_with_logits(
                subclass_logits[mask][:, 0],  # take first logit as class1 score
                subclass_binary_labels,
            )
            loss += subclass_loss

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        X, y = batch["embed"], batch["label"]
        logits = self(X)
        loss = F.cross_entropy(logits, y)

        # to prevent the warning: Trying to infer the `batch_size` from an ambiguous collection
        # add the argument: batch_size=len(batch)
        self.log("val/loss", loss, batch_size=len(batch))

        # metrics calculation
        probs = self.softmax(logits)

        # self.pr_curve.update(probs, y)
        self.macro_ap.update(probs, y)
        self.ap.update(probs, y)
        self.prec.update(probs, y)
        self.recall.update(probs, y)

    def on_validation_epoch_end(self):
        # macro means directly average all average precision from different classes
        self.log("val/macro_avg_prec", self.macro_ap.compute())

        # log individual average precision, precision, recall
        aps = self.ap.compute()
        precs = self.prec.compute()
        recalls = self.recall.compute()

        for class_type, ap, prec, recall in zip(self.classes, aps, precs, recalls):
            self.log(f"val/ap_{class_type}", ap)
            self.log(f"val/prec_{class_type}", prec)
            self.log(f"val/recall_{class_type}", recall)

    def test_step(self, batch, batch_idx) -> None:
        X, y = batch["embed"], batch["label"]

        # metrics calculation
        probs = self.softmax(self(X))
        # self.pr_curve.update(probs, y)

        self.macro_ap.update(probs, y)
        self.ap.update(probs, y)
        self.prec.update(probs, y)
        self.recall.update(probs, y)

        self.macro_f1.update(probs, y)
        self.f1.update(probs, y)

        self.macro_auroc.update(probs, y)
        self.auroc.update(probs, y)

    def on_test_epoch_end(self):
        # log precision recall curves (table)
        # prec, recall, thres = self.pr_curve.compute()
        # self.metrics_logger.log_pr_curves("test", prec, recall, thres)

        self.log("test/macro_avg_prec", self.macro_ap.compute())
        self.log("test/macro_f1", self.macro_f1.compute())
        self.log("test/macro_auroc", self.macro_auroc.compute())

        # log individual average precision, precision, recall
        aps = self.ap.compute()
        precs = self.prec.compute()
        recalls = self.recall.compute()
        f1s = self.f1.compute()
        aurocs = self.auroc.compute()

        for class_type, ap, prec, recall, f1, auroc in zip(
            self.classes, aps, precs, recalls, f1s, aurocs
        ):
            self.log(f"test/ap_{class_type}", ap)
            self.log(f"test/prec_{class_type}", prec)
            self.log(f"test/recall_{class_type}", recall)

            self.log(f"test/f1_{class_type}", f1)
            self.log(f"test/auroc_{class_type}", auroc)
