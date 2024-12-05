####################################################################################################
#
####################################################################################################
import hydra
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig
import optuna

from core.lightning_module import HawkishDovishClassifier
from train import setup_dm, setup_trainer


class Tuner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def objective(self, trial):
        batch_size = trial.suggest_int("batch_size", 256, 1536)
        dm = setup_dm(self.cfg, batch_size)

        nn_hparam = {
            "batch_size": batch_size,
            "embed_model_name": dm.embed_model_name,
        }

        if self.cfg.pooling_strategy == "rnn":
            nn_hparam = (
                self.rnn_hparam(trial)
                | {
                    "flair_layers": dm.flair_layers,
                }
                | self.ff_hparam(trial)
                | nn_hparam
            )
        elif self.cfg.pooling_strategy in ["cls_pooler", "last_layer_mean_pooler"]:
            nn_hparam = self.ff_hparam(trial) | nn_hparam

        lr = trial.suggest_float("lr", 3e-6, 3e-3)

        model = HawkishDovishClassifier(
            self.cfg.pooling_strategy,
            self.cfg.nn,
            lr,
            dm.sklearn_class_weight,
            dm.embed_dimension,
            **nn_hparam,
        )

        logger = MLFlowLogger(
            experiment_name=self.cfg.tuning.study_name,
            run_name=str(trial.number),
            tracking_uri="file:./mlruns",
            log_model=True,
        )
        self.trainer = setup_trainer(self.cfg, logger)
        self.trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

        return self.trainer.checkpoint_callback.best_model_score.item()

    def optimize(self):
        study = optuna.create_study(
            storage="sqlite:///db.sqlite3",
            study_name=self.cfg.tuning.study_name,
            pruner=optuna.pruners.MedianPruner(),
            load_if_exists=True,
            direction=(
                "maximize" if self.cfg.model_check_point.mode == "max" else "minimize"
            ),
        )
        study.optimize(self.objective, n_trials=self.cfg.tuning.n_trials)

    def rnn_hparam(self, trial: optuna.Trial):
        hparam = {
            "hidden_size": trial.suggest_int("hidden_size", 256, 2048),
            # "num_layers": trial.suggest_int("num_layers", 1, 20),
            "num_layers": 1,
            # "dropout": trial.suggest_float("dropout", 0.1, 0.9)
            "dropout": 0.0,
            "bidirectional": True,
        }

        return hparam

    def ff_hparam(self, trial: optuna.Trial):
        hparam = {
            "ff_dropout": trial.suggest_float("ff_dropout", 0.2, 0.7),
        }

        return hparam


@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    tuner = Tuner(cfg)
    tuner.optimize()


if __name__ == "__main__":
    main()
