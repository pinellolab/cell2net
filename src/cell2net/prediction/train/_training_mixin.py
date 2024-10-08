from lightning import LightningDataModule

from ._training_plan import TrainingPlan
from ._training_runner import TrainRunner


class UnsupervisedTrainingMixin:
    """General purpose unsupervised train method."""

    # @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 20,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: int = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        datamodule: LightningDataModule | None = None,
        **trainer_kwargs,
    ):

        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.train_size = train_size
        self.validation_size = (validation_size,)
        self.shuffle_set_split = shuffle_set_split
        self.load_sparse_tensor = load_sparse_tensor
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.datasplitter_kwargs = datasplitter_kwargs
        self.plan_kwargs = plan_kwargs
        self.datamodule = datamodule

        plan_kwargs = plan_kwargs or {}
        training_plan = TrainingPlan(self.module, **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=datamodule,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )

        return runner()
