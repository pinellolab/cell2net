from abc import abstractmethod

import pandas as pd
import torch


class BaseModel:
    def __init__(self):
        self.history_ = None
        self.is_trained_ = False
        self.summary_ = ""
        self.device_ = ""

    @property
    def summary(self):
        return self.summary_

    @summary.setter
    def summary(self, value):
        self.summary_ = value

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self.is_trained_

    @is_trained.setter
    def is_trained(self, value):
        self.is_trained_ = value

    @property
    def history(self) -> None | pd.DataFrame:
        return self.history_

    def to_device(self, device_name: str | int):
        """Move model to device.

        Parameters
        ----------
        device
            Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
            or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.
        """
        self.device_ = torch.device(device_name)
        self.module.to(self.device_)  # type: ignore

    @property
    def device(self) -> str:
        """The current device that the module's params are on."""
        return self.device_  # type: ignore

    @device.setter
    def device(self, value):
        self.device_ = value

    @abstractmethod
    def train(self):
        """Trains the model."""
