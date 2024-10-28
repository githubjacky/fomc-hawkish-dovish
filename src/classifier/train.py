####################################################################################################
# This file integrate several auxiliary components for training a classifier, such as equipping a 
# logger, how to save model checkpoints, when to early stop and configuring the progress bar, etc. 
# Moreover, the file serves as the entry point to lauch the training procedure, incoporating the
# data module with stacked nn architectures.
####################################################################################################


import hydra
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from data_module import RNNFamilyDataModule, MLPDataModule
from model import RNNFamily, HawkishDovishClassifier


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def train(cfg: DictConfig):
    """
    To train a classifier, we need to set up data module, `model` and `trainer`. After setting up
    we can fit the model and then test the model.

    - `model` is the object of the L.LightningModule's subclass` defind in model.py
    """
    # task 1: setup data module
    dm = (
        # `RNNFamily` can only use flair's word embeddings
        RNNFamilyDataModule(
            cfg.batch_size,
            cfg.flair_embed.model_name, 
            cfg.flair_embed.flair_layers,
            cfg.flair_embed.flair_layer_mean
        )
        if cfg.nn != "MLP"
        else
        MLPDataModule(
            cfg.batch_size,
            cfg.embed_framework,
            (
                cfg.flair_embed.model_name
                if cfg.embed_framework == "flair"
                else
                cfg.sbert_embed.model_name
            ), 
            # if we're using sbert, then it's meaningless to set these arguments
            # but I keep them just in case we're using flair
            cfg.flair_embed.flair_layers,
            cfg.flair_embed.flair_layer_mean
        )
    )
    dm.prepare_data()

    # task2: set up `model` for `trainer``
    train_idx, val_idx = train_test_split(
        np.arange(len(dm.train_dataset)),
        test_size=0.2,
        stratify=dm.train_dataset["label"],
        random_state=cfg.random_state
    )
    dm.setup(train_idx, val_idx)

    # task 2 setup nn
    nn = (
        RNNFamily(
            cfg.nn,
            input_size = dm.embed_dimension,
            hidden_size = cfg.RNNFamily.hidden_size,
            num_layers = cfg.RNNFamily.num_layers,
            dropout = cfg.RNNFamily.dropout,
            bidirectional = cfg.RNNFamily.bidirectional
        )
        if cfg.nn != "MLP"
        else
        # TODO
        None
    )
    model = HawkishDovishClassifier(cfg, nn, dm.sklearn_class_weight)

    # task3: configure the trainer
    wandb_logger = WandbLogger(
        project = "fomc-hawkish-dovish",
        # log model once the checkpoint is created
        log_model = "all",
        group = cfg.tuning.study_name
    )
    wandb_logger.watch(model, log="all")

    early_stop = EarlyStopping(
        monitor = cfg.early_stop.monitor,
        mode = cfg.early_stop.mode,
        patience = cfg.early_stop.patience
    )
    ckpt = ModelCheckpoint(
        monitor = cfg.model_check_point.monitor,
        mode = cfg.model_check_point.mode
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks=[early_stop, ckpt, pbar]

    trainer = L.Trainer(
        accelerator = cfg.trainer.accelerator,
        strategy = cfg.trainer.strategy,
        devices = cfg.trainer.devices,
        # num_nodes = 1,
        # precision = "32-true"
        logger = wandb_logger,
        callbacks = callbacks,
        # fast_dev_run = True
        max_epochs = cfg.trainer.max_epochs,
        log_every_n_steps = cfg.trainer.log_every_n_steps ,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch,
        # accumulate_grad_batches =
        # gradient_clip_val = 
    )
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    trainer.test(dataloaders=dm.test_dataloader(), ckpt_path='best')


if __name__ == "__main__":
    train()