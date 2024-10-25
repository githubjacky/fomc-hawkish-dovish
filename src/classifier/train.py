####################################################################################################
#
####################################################################################################


import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from data_module import RNNFamilyDataModule, MLPDataModule
from model import RNNFamily, HawkishDovishClassifier


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    """
    To train a classifier, we need to set up `trainer `, data module and model. After setting up
    we can fit the model and then test the model.
    """
    ckpt = ModelCheckpoint(
        dirpath = "model_ckpt/rnn/",
        filename='{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        enable_version_counter = False
    )
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks=[ckpt, pbar]

    trainer = L.Trainer(
        max_epochs = cfg.trainer.max_epochs, 
        callbacks = callbacks, 
        log_every_n_steps = cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch,
        accelerator = "gpu",
        devices = [0]
    )

    dm = (
        RNNFamilyDataModule()
        if cfg.model != "MLP"
        else
        MLPDataModule()
    )
    dm.prepare_data()

    train_idx, val_idx = train_test_split(
        dm.train_dataset["label"],
        test_size=0.2,
        stratify=dm.train_dataset["label"],
        random_state=cfg.random_state
    )
    dm.setup(train_idx, val_idx)

    # TODO: calculate the dimension of embeddings in dm
    rnn = RNNFamily(
        "RNN",
        input_size = 768,
        hidden_size = cfg.RNNFamily.hidden_size,
        num_layers = cfg.RNNFamily.num_layers,
        dropout = cfg.RNNFamily.dropout,
        bidirectional = cfg.RNNFamily.bidirectional
    )
    classifier = HawkishDovishClassifier(cfg, rnn)
    trainer.fit(classifier, dm.train_dataloader(), dm.val_dataloader())
    trainer.test(classifier, dataloaders=dm.test_dataloader())


if __name__ == "__main__":
    main()