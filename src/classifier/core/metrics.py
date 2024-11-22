from typing import Literal


class ClassificationMetricsLogger:
    # logger is the `L.LightningModule`'s logger
    def __init__(self, logger):
        self.num_classes = 3
        self.logger = logger

        self.classes = ["dovish", "hawkish", "neutral"]

    def log_pr_curves(self, stage: Literal["val", "test"], prec, recall, thres):
        for i in range(self.num_classes):
            class_prec = prec[i].cpu().numpy()
            class_recall = recall[i].cpu().numpy()
            class_thres = thres[i].cpu().numpy()

            data = [[x, y, z] for x, y, z in zip(class_recall, class_prec, class_thres)]
            columns = ["recall", "precision", "threshold"]

        self.logger.experiment.log_table(
            key=f"{stage}/pr_curve_{self.classes[i]}", columns=columns, data=data
        )

