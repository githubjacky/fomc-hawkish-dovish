##########################################################################################
# This file integrate several auxiliary components for training a classifier, such as
# equipping a logger, how to save model checkpoints, when to early stop and configuring
# the progress bar, etc. Moreover, the file serves as the entry point to launch the
# training procedure, incorporating the data module with stacked nn architectures.
#
# To train a classifier, we need to set up data module, `model` and `trainer`. After
# setting up, we can fit the model and then test the model.
#
# Note:
# - `model` is the object of the L.LightningModule's subclass` defind in model.py
##########################################################################################


import hydra
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import optuna
from sklearn.model_selection import train_test_split
from typing import Optional
import wandb

from core.data_module import RNNFamilyDataModule, MLPDataModule
from core.lightning_module import HawkishDovishClassifier


def setup_dm(cfg: DictConfig, batch_size: Optional[int] = None):
    # not in hyperparmeter tuning
    batch_size = cfg.batch_size if batch_size is None else batch_size

    # `RNNFamily` can only use flair's word embeddings
    if cfg.nn in ["RNN", "GRU", "LSTM"]:
        dm = RNNFamilyDataModule(
            batch_size,
            cfg.flair_embed.model_name,
            cfg.flair_embed.flair_layers,
            cfg.flair_embed.flair_layer_mean,
        )
    else:
        dm = MLPDataModule(
            batch_size,
            cfg.embed_framework,
            (
                cfg.flair_embed.model_name
                if cfg.embed_framework == "flair"
                else cfg.sbert_embed.model_name
            ),
            # if we're using sbert, then it's meaningless to set these arguments
            # but I keep them just in case we're using flair
            cfg.flair_embed.flair_layers,
            cfg.flair_embed.flair_layer_mean,
        )

    dm.prepare_data()

    train_idx, val_idx = train_test_split(
        np.arange(len(dm.train_dataset)),
        test_size=0.2,
        stratify=dm.train_dataset["label"],
        random_state=cfg.random_state,
    )
    dm.setup(train_idx, val_idx)

    return dm


def setup_model(dm, cfg: DictConfig):
    if cfg.nn in ["RNN", "GRU", "LSTM"]:
        nn_hparam = OmegaConf.to_container(cfg.RNNFamily) | OmegaConf.to_container(
            cfg.ff
        )

    model = HawkishDovishClassifier(
        cfg.nn, cfg.lr, dm.sklearn_class_weight, dm.embed_dimension, **nn_hparam
    )

    return model


def setup_trainer(model, cfg: DictConfig):
    wandb_logger = WandbLogger(
        project="fomc-hawkish-dovish",
        # log model once the checkpoint is created
        log_model="all",
        group=cfg.tuning.study_name,
        save_dir="wandb",
    )
    wandb_logger.watch(model, log="all")

    early_stop = EarlyStopping(
        monitor=cfg.early_stop.monitor,
        mode=cfg.early_stop.mode,
        patience=cfg.early_stop.patience,
    )
    ckpt = ModelCheckpoint(
        monitor=cfg.model_check_point.monitor, mode=cfg.model_check_point.mode
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [early_stop, ckpt, pbar]

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        # num_nodes = 1,
        # precision = "32-true"
        logger=wandb_logger,
        callbacks=callbacks,
        # fast_dev_run = True
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        # accumulate_grad_batches =
        # gradient_clip_val =
    )

    return trainer


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    dm = setup_dm(cfg)
    model = setup_model(dm, cfg)
    trainer = setup_trainer(model, cfg)

    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    trainer.test(dataloaders=dm.test_dataloader(), ckpt_path="best")


if __name__ == "__main__":
    main()
