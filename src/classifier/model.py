####################################################################################################
# This file defines how the traing process should be proceed, such as what training loos is
# considered, what metris are logged during validation and test stages, and what optimizers should 
# we adopt. `L.LightningModule` chunks the process into different pieces of events to allow easy
# configurations separately.
####################################################################################################


from functools import cached_property
import lightning as L
from omegaconf import DictConfig
import torch
from torch.nn import Linear, Softmax
import torch.nn.functional as F
from torchmetrics.classification import PrecisionRecallCurve, AveragePrecision, Precision, Recall
from typing import Optional


from metrics import ClassificationMetricsLogger
from nn import RNNFamily


class HawkishDovishClassifier(L.LightningModule):
    def __init__(self, hparam: DictConfig, model, class_weights: Optional[torch.Tensor] = None):
        """
        The type of `model` is the subclass of `nn.Module`, i.e., nn.RNN.
        """
        super().__init__()
        # avoid: Attribute 'model' is an instance of `nn.Module` and is already saved during 
        # checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['model'])`.
        self.save_hyperparameters(ignore=['model'])  # save the input arguments to wandb
        self.hparam = hparam
        self.model = model
        self.clas_weights = class_weights

        # 3 is the number of classes
        if type(model) == RNNFamily:
            hidden_size = (
                # concat the first and the last hidden state, both's size are double as each of
                # which concate the output from two directions.
                hparam.RNNFamily.hidden_size * 4
                if hparam.RNNFamily.bidirectional
                else
                # only use the last hidden state
                hparam.RNNFamily.hidden_size
            )
        else:
            hidden_size = hparam.MLP.output_size
    
        self.linear = Linear(hidden_size, 3)
        self.softmax = Softmax(dim=-1)

        # validation and test metrics
        self.pr_curve = PrecisionRecallCurve("multiclass", num_classes=3)
        self.macro_ap = AveragePrecision("multiclass", num_classes=3, average="macro")
        self.ap = AveragePrecision("multiclass", num_classes=3, average=None)
        self.prec = Precision("multiclass", num_classes=3, average=None)
        self.recall = Recall("multiclass", num_classes=3, average=None)


    @cached_property
    def classes(self):
        return ["dovish", "hawkish", "neutral"]


    @cached_property
    def metrics_logger(self):
        return ClassificationMetricsLogger(self.logger)


    def forward(self, X):
        outputs = self.model(X)
        logits = self.linear(outputs)

        return logits


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparam.lr)


    def training_step(self, batch, batch_idx):
        X, y = batch['embed'], batch['label']
        loss = F.cross_entropy(self(X), y, weight=torch.tensor([2., 2., 1.]).cuda())
        self.log("train/loss", loss)

        return loss


    def validation_step(self, batch, batch_idx) -> None:
        X, y = batch['embed'], batch['label']
        logits = self(X)
        loss = F.cross_entropy(logits, y)

        # to prevent the warning: Trying to infer the `batch_size` from an ambiguous collection
        # add the argument: batch_size=len(batch)
        self.log("val/loss", loss, batch_size=len(batch))

        # metrics calculation
        probs = self.softmax(logits)

        self.pr_curve.update(probs, y)
        self.macro_ap.update(probs, y)
        self.ap.update(probs, y)
        self.prec.update(probs, y)
        self.recall.update(probs, y)
        

    def on_validation_epoch_end(self):
        # log precision recall curves (table)
        prec, recall, thres = self.pr_curve.compute()
        self.metrics_logger.log_pr_curves("val", prec, recall, thres)

        # macro means directly average all average precision from different classes
        self.log("val/macro_avg_prec", self.macro_ap)

        # log individual average precision, precision, recall
        aps = self.ap.compute()
        precs = self.prec.compute()
        recalls = self.recall.compute()

        for class_type, ap, prec, recall in zip(self.classes, aps, precs, recalls):
            self.log(f"val/ap_{class_type}", ap)
            self.log(f"val/prec_{class_type}", prec)
            self.log(f"val/recall_{class_type}", recall)


    def test_step(self, batch, batch_idx) -> None:
        X, y = batch['embed'], batch['label']

        # metrics calculation
        probs = self.softmax(self(X))
        self.pr_curve.update(probs, y)

        self.macro_ap.update(probs, y)
        self.ap.update(probs, y)
        self.prec.update(probs, y)
        self.recall.update(probs, y)


    def on_test_epoch_end(self):
        # log precision recall curves (table)
        prec, recall, thres = self.pr_curve.compute()
        self.metrics_logger.log_pr_curves("test", prec, recall, thres)

        self.log("test/macro_avg_prec", self.macro_ap)

        # log individual average precision, precision, recall
        aps = self.ap.compute()
        precs = self.prec.compute()
        recalls = self.recall.compute()

        for class_type, ap, prec, recall in zip(self.classes, aps, precs, recalls):
            self.log(f"test/ap_{class_type}", ap)
            self.log(f"test/prec_{class_type}", prec)
            self.log(f"test/recall_{class_type}", recall)