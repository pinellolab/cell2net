from typing import Literal

import lightning as pl
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrainingPlan(pl.LightningModule):
    def __init__(
        self,
        module: nn.Module,
        optimizer: Literal["Adam", "AdamW"] = "Adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        reduce_lr_on_plateau: bool = True,
        lr_factor: float = 0.5,
        lr_patience: int = 5,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: str = "validation_loss",
        lr_min: float = 1e-5,
    ) -> None:
        super().__init__()

        self.module = module
        self.optimizer_name = optimizer

        self.lr = lr
        self.weight_decay = weight_decay
        self.reduce_lr_on_plateau = (reduce_lr_on_plateau,)
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_threshold = lr_threshold
        self.lr_scheduler_metric = lr_scheduler_metric
        self.lr_min = lr_min

        self.train_loss = 0
        self.valid_loss = 0

    def training_step(self, batch, batch_idx):
        """Training step for the model."""
        self.forward(batch)

        # return scvi_loss.loss

    def configure_optimizers(self):
        """Configure optimizers for the model."""
        params = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer = self.get_optimizer_creator()(params)
        config = {"optimizer": optimizer}

        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )
        return config
