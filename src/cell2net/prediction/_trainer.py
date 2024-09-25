from typing import Literal

import lightning as pl
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import Logger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ._logger import SimpleLogger


class Trainer(pl.Trainer):
    def __init__(
        self,
        *,
        accelerator: str | Accelerator = "auto",
        devices: list[int] | str | int = "auto",
        benchmark: bool = True,
        check_val_every_n_epoch: int | None = None,
        max_epochs: int = 20,
        default_root_dir: str | None = None,
        enable_checkpointing: bool = False,
        checkpointing_monitor: str = "validation_loss",
        num_sanity_val_steps: int = 0,
        enable_model_summary: bool = False,
        early_stopping: bool = False,
        early_stopping_monitor: str = "validation_loss",
        early_stopping_min_delta: float = 0.00,
        early_stopping_patience: int = 10,
        early_stopping_mode: Literal["min", "max"] = "min",
        enable_progress_bar: bool = True,
        logger: Logger | None | bool = None,
        progress_bar_refresh_rate: int = 1,
        simple_progress_bar: bool = True,
        # logger: Logger | None | bool = None,
        log_every_n_steps: int = 10,
        learning_rate_monitor: bool = False,
        **kwargs,
    ) -> None:
        self._model = None

        # if default_root_dir is None:
        #     default_root_dir = settings.logging_dir

        check_val_every_n_epoch = check_val_every_n_epoch
        callbacks = kwargs.pop("callbacks", [])

        if early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor=early_stopping_monitor,
                min_delta=early_stopping_min_delta,
                patience=early_stopping_patience,
                mode=early_stopping_mode,
            )
            callbacks.append(early_stopping_callback)
            check_val_every_n_epoch = 1

        if learning_rate_monitor and not any(isinstance(c, LearningRateMonitor) for c in callbacks):
            callbacks.append(LearningRateMonitor())
            check_val_every_n_epoch = 1

        if logger is None:
            logger = SimpleLogger()

        super().__init__(
            accelerator=accelerator,
            devices=devices,
            benchmark=benchmark,
            check_val_every_n_epoch=check_val_every_n_epoch,
            max_epochs=max_epochs,
            default_root_dir=default_root_dir,
            enable_checkpointing=enable_checkpointing,
            num_sanity_val_steps=num_sanity_val_steps,
            enable_model_summary=enable_model_summary,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=enable_progress_bar,
            callbacks=callbacks,
            **kwargs,
        )

    def fit(self, train_dataloaders, val_dataloaders):
        """Fit the model."""
        return NotImplementedError

        # super().fit(
        #     model=self._model,
        #     train_dataloaders=train_dataloaders,
        #     val_dataloaders=val_dataloaders,
        # )


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


# class TrainRunner:
#     """TrainRunner calls Trainer.fit() and handles pre and post training procedures."""

#     def __init__(
#         self,
#         model: BaseModelClass,
#         max_epochs: int,
#         data_splitter: DataSplitter,
#         training_plan: pl.LightningModule,
#         accelerator: str = "auto",
#         devices: int | list[int] | str = "auto",
#         **trainer_kwargs,
#     ) -> None:
#         self.model = model
#         self.training_plan = training_plan
#         self.max_epochs = max_epochs
#         self.data_splitter = data_splitter
#         self.accelerator, lightning_devices, device = parse_device_args(
#             accelerator=accelerator,
#             devices=devices,
#             return_device="torch",
#         )
#         self.accelerator = accelerator
#         self.lightning_devices = lightning_devices
#         self.device = device

#         if getattr(self.training_plan, "reduce_lr_on_plateau", False):
#             trainer_kwargs["learning_rate_monitor"] = True

#         self.trainer = Trainer(
#             max_epochs=self.max_epochs,
#             accelerator=self.accelerator,
#             devices=self.lightning_devices,
#         )
#         self.trainer.model_ = model

#     def __call__(self):
#         """Run training."""
#         if hasattr(self.data_splitter, "n_train"):
#             self.training_plan.n_obs_training = self.data_splitter.n_train
#         if hasattr(self.data_splitter, "n_val"):
#             self.training_plan.n_obs_validation = self.data_splitter.n_val

#         self.trainer.fit(self.training_plan, self.data_splitter)
#         self._update_history()

#         # data splitter only gets these attrs after fit
#         self.model.train_indices = getattr(self.data_splitter, "train_idx", None)
#         self.model.test_indices = getattr(self.data_splitter, "test_idx", None)
#         self.model.validation_indices = getattr(self.data_splitter, "val_idx", None)

#         self.model.module.eval()
#         self.model.is_trained_ = True
#         self.model.to_device(self.device)
#         self.model.trainer = self.trainer

#     def _update_history(self):
#         return NotImplemented


# class UnsupervisedTrainingMixin:
#     """General purpose unsupervised train method."""

#     @devices_dsp.dedent
#     def train(
#         self,
#         max_epochs: int = 20,
#         accelerator: str = "auto",
#         devices: int | list[int] | str = "auto",
#         train_size: float = 0.9,
#         validation_size: float | None = None,
#         shuffle_set_split: bool = True,
#         load_sparse_tensor: bool = False,
#         batch_size: int = 128,
#         early_stopping: bool = False,
#         datasplitter_kwargs: dict | None = None,
#         plan_kwargs: dict | None = None,
#         datamodule: LightningDataModule | None = None,
#         **trainer_kwargs,
#     ):

#         self.max_epochs = max_epochs
#         self.accelerator = accelerator
#         self.devices = devices
#         self.train_size = train_size
#         self.validation_size = (validation_size,)
#         self.shuffle_set_split = shuffle_set_split
#         self.load_sparse_tensor = load_sparse_tensor
#         self.batch_size = batch_size
#         self.early_stopping = early_stopping
#         self.datasplitter_kwargs = datasplitter_kwargs
#         self.plan_kwargs = plan_kwargs
#         self.datamodule = datamodule

#         plan_kwargs = plan_kwargs or {}
#         training_plan = TrainingPlan(self.module, **plan_kwargs)

#         es = "early_stopping"
#         trainer_kwargs[es] = (
#             early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
#         )

#         runner = TrainRunner(
#             self,
#             training_plan=training_plan,
#             data_splitter=datamodule,
#             max_epochs=max_epochs,
#             accelerator=accelerator,
#             devices=devices,
#             **trainer_kwargs,
#         )

#         return runner()
