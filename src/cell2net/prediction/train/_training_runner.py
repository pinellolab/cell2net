import lightning as pl
from scvi.train import Trainer
from torch import nn

from cell2net.prediction.data import DataSplitter

from ._utils import parse_device_args


class TrainRunner:
    """TrainRunner calls Trainer.fit() and handles pre and post training procedures."""

    def __init__(
        self,
        model: nn.Module,
        max_epochs: int,
        data_splitter: DataSplitter,
        training_plan: pl.LightningModule,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        **trainer_kwargs,
    ) -> None:
        self.model = model
        self.training_plan = training_plan
        self.max_epochs = max_epochs
        self.data_splitter = data_splitter
        self.accelerator, lightning_devices, device = parse_device_args(
            accelerator=accelerator,
            devices=devices,
            return_device="torch",
        )
        self.accelerator = accelerator
        self.lightning_devices = lightning_devices
        self.device = device

        if getattr(self.training_plan, "reduce_lr_on_plateau", False):
            trainer_kwargs["learning_rate_monitor"] = True

        self.trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.lightning_devices,
        )
        self.trainer.model_ = model

    def __call__(self):
        """Run training."""
        if hasattr(self.data_splitter, "n_train"):
            self.training_plan.n_obs_training = self.data_splitter.n_train
        if hasattr(self.data_splitter, "n_val"):
            self.training_plan.n_obs_validation = self.data_splitter.n_val

        self.trainer.fit(self.training_plan, self.data_splitter)
        self._update_history()

        # data splitter only gets these attrs after fit
        self.model.train_indices = getattr(self.data_splitter, "train_idx", None)
        self.model.test_indices = getattr(self.data_splitter, "test_idx", None)
        self.model.validation_indices = getattr(self.data_splitter, "val_idx", None)

        self.model.module.eval()
        self.model.is_trained_ = True
        self.model.to_device(self.device)
        self.model.trainer = self.trainer

    def _update_history(self):
        return NotImplemented
