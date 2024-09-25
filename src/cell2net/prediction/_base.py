import os
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from ._constants import SAVE_KEYS


class BaseModelClass(metaclass=ABCMeta):
    def __init__(
        self,
        module: nn.Module,
        is_trained: bool = False,
        model_summary_string: str = "",
        train_indices: list | None = None,
        validation_indices: list | None = None,
        test_indices: list | None = None,
        history: str | None = None,
    ) -> None:
        self.module = module
        self.is_trained_ = is_trained
        self._model_summary_string = model_summary_string
        self.train_indices_ = train_indices
        self.validation_indices_ = validation_indices
        self.test_indices_ = test_indices
        self.history_ = history

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self.is_trained_

    @is_trained.setter
    def is_train(self, value: bool):
        self.is_trained_ = value

    @property
    def train_indices(self):
        """Observations that are in train set."""
        return self.train_indices_

    @property
    def validation_indices(self):
        """Observations that are in validation set."""
        return self.validation_indices_

    @property
    def test_indices(self):
        """Observations that are in test set."""
        return self.test_indices_

    @train_indices.setter
    def train_indices(self, value):
        self.train_indices_ = value

    @validation_indices.setter
    def validation_indices(self, value):
        self.validation_indices_ = value

    @test_indices.setter
    def test_indices(self, value):
        self.test_indices_ = value

    @property
    def history(self):
        return self.history_

    def to_device(self, device: str | int):
        """Move model to device.

        Parameters
        ----------
        device
            Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
            or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.
        """
        my_device = torch.device(device)
        self.module.to(my_device)

    @property
    def device(self) -> str:
        """The current device that the module's params are on."""
        return self.module.device

    def save(self, dir_path: str, prefix: str):
        """Save the state of the model"""
        model_save_path = os.path.join(dir_path, f"{prefix}")

        # save the model state dict and the trainer state dict only
        model_state_dict = self.module.state_dict()

        torch.save(
            {
                SAVE_KEYS.MODEL_STATE_DICT_KEY: model_state_dict,
            },
            model_save_path,
        )

    # def load(
    #     self,
    #     dir_path: str,
    #     prefix: str | None = None,
    #     map_location: Literal["cpu", "cuda"] | None = None,
    #     accelerator: str = "auto",
    #     device: int | list[int] | str = "auto",
    #     backup_url: str | None = None,
    # ):
    #     """Instantiate a model from the saved output."""
    #     model_path = os.path.join(dir_path, f"{prefix}")

    #     model = torch.load(model_path, map_location=map_location)

    #     self.module =

    @abstractmethod
    def train(self):
        """Trains the model."""
