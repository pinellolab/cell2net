import os
from abc import ABCMeta, abstractmethod

import torch

from cell2net.prediction.data import MuDataManager

from ._constants import SAVE_KEYS

_SETUP_INPUTS_EXCLUDED_PARAMS = {"adata", "mdata", "kwargs"}


class BaseModelMetaClass(ABCMeta):
    """Metaclass for :class:`~scvi.model.base.BaseModelClass`.

    Constructs model class-specific mappings for :class:`~scvi.data.AnnDataManager` instances.
    ``cls._setup_adata_manager_store`` maps from AnnData object UUIDs to
    :class:`~scvi.data.AnnDataManager` instances.

    This mapping is populated everytime ``cls.setup_anndata()`` is called.
    ``cls._per_isntance_manager_store`` maps from model instance UUIDs to AnnData UUID:
    :class:`~scvi.data.AnnDataManager` mappings.
    These :class:`~scvi.data.AnnDataManager` instances are tied to a single model instance and
    populated either
    during model initialization or after running ``self._validate_anndata()``.
    """

    @abstractmethod
    def __init__(cls, name, bases, dct):
        cls._setup_mdata_manager_store: dict[str, type[MuDataManager]] = (
            {}
        )  # Maps adata id to AnnDataManager instances.
        cls._per_instance_manager_store: dict[str, dict[str, type[MuDataManager]]] = (
            {}
        )  # Maps model instance id to AnnDataManager mappings.
        super().__init__(name, bases, dct)


class BaseModelClass(metaclass=BaseModelMetaClass):
    def __init__(
        self,
    ) -> None:
        self.module = None
        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.validation_indices_ = None
        self.test_indices_ = None
        self.history_ = None

    @property
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        return self.is_trained_

    @is_trained.setter
    def is_train(self, value: bool):
        self.is_trained_ = value

    def to_device(self, device: str | int):
        """Move model to device.

        Parameters
        ----------
        device
            Device to move model to. Options: 'cpu' for CPU, integer GPU index (eg. 0),
            or 'cuda:X' where X is the GPU index (eg. 'cuda:0'). See torch.device for more info.
        """
        my_device = torch.device(device)
        self.module.to(my_device)  # type: ignore

    @property
    def device(self) -> str:
        """The current device that the module's params are on."""
        return self.module.device  # type: ignore

    def save(self, dir_path: str, prefix: str):
        """Save the state of the model"""
        model_save_path = os.path.join(dir_path, f"{prefix}")

        # save the model state dict and the trainer state dict only
        model_state_dict = self.module.state_dict()  # type: ignore

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
