####################################################################################################
#
####################################################################################################
import hydra
from omegaconf import DictConfig
import optuna
import wandb

from model import RNNFamily, HawkishDovishClassifier
from train import setup_dm, setup_trainer


class Tuner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def objective(self, trial):
        batch_size = trial.suggest_int("batch_size", 32, 256)
        lr = trial.suggest_float("lr", 3e-6, 0.1)
        dm = setup_dm(self.cfg, batch_size)

        if self.cfg.nn in ["RNN", "GRU", "LSTM"]:
            nn_hparam = self.rnn_hparam(trial)
            nn = RNNFamily(self.cfg.nn, input_size = dm.embed_dimension, **nn_hparam)

        model = HawkishDovishClassifier(nn, lr, dm.sklearn_class_weight, **nn_hparam)
        trainer = setup_trainer(model, self.cfg)
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
        wandb.finish()
        
        return trainer.checkpoint_callback.best_model_score.item()


    def optimize(self):
        study = optuna.create_study(
            storage = "sqlite:///db.sqlite3", 
            study_name = self.cfg.tuning.study_name,
            pruner = optuna.pruners.MedianPruner(),
            load_if_exists=True,
            direction = "maximize" if self.cfg.model_check_point.mode == "max" else "minimize"
        )
        study.optimize(self.objective, n_trials=self.cfg.tuning.n_trials)


    def rnn_hparam(self, trial: optuna.Trial):
        hparam = {
            "hidden_size": trial.suggest_int("hidden_size", 64, 256),
            "num_layers": trial.suggest_int("num_layers", 1, 20),
            "dropout": trial.suggest_float("dropout", 0.1, 0.9)
        }
        hparam = hparam | {"bidirectional": self.cfg.RNNFamily.bidirectional}

        return hparam
    

@hydra.main(config_path="../../config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    tuner = Tuner(cfg)
    tuner.optimize()


if __name__ == "__main__":
    main()