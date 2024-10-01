import os
from abc import ABCMeta, abstractmethod

import torch
from mudata import MuData

from cell2net.prediction.data import MuDataManager
from cell2net.prediction.data._constants import (
    _MODEL_NAME_KEY,
    _SETUP_ARGS_KEY,
    _SETUP_METHOD_NAME,
)

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
        mdata: MuData,
    ) -> None:
        self.module = None
        self._mdata = mdata
        self.is_trained_ = False
        self._model_summary_string = ""
        self.train_indices_ = None
        self.validation_indices_ = None
        self.test_indices_ = None
        self.history_ = None

    @property
    def mdata(self) -> MuData:
        """Data attached to model instance."""
        return self._mdata

    @mdata.setter
    def mdata(self, mdata: MuData):
        if mdata is None:
            raise ValueError("mdata cannot be None.")
        self._validate_mudata(mdata)
        self._mdata = mdata
        # self._mdata_manager = self.get_mudata_manager(mdata, required=True)
        # self.registry_ = self._adata_manager.registry
        # self.summary_stats = self._adata_manager.summary_stats

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

    @staticmethod
    def _get_setup_method_args(**setup_locals) -> dict:
        """Returns a dictionary organizing the arguments used to call ``setup_anndata``.

        Must be called with ``**locals()`` at the start of the ``setup_anndata`` method
        to avoid the inclusion of any extraneous variables.
        """
        cls = setup_locals.pop("cls")
        method_name = None
        if "adata" in setup_locals:
            method_name = "setup_anndata"
        elif "mdata" in setup_locals:
            method_name = "setup_mudata"

        model_name = cls.__name__
        setup_args = {}
        for k, v in setup_locals.items():
            if k not in _SETUP_INPUTS_EXCLUDED_PARAMS:
                setup_args[k] = v
        return {
            _MODEL_NAME_KEY: model_name,
            _SETUP_METHOD_NAME: method_name,
            _SETUP_ARGS_KEY: setup_args,
        }

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

    def _validate_mudata(self, mdata: MuData):
        if mdata is None:
            mdata = self.mdata

        return mdata

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

    @classmethod
    def register_manager(cls, mudata_manager: MuDataManager):
        """Registers an :class:`~scvi.data.AnnDataManager` instance with this model class.

        Stores the :class:`~scvi.data.AnnDataManager` reference in a class-specific manager store.
        Intended for use in the ``setup_anndata()`` class method followed up by retrieval of the
        :class:`~scvi.data.AnnDataManager` via the ``_get_most_recent_anndata_manager()`` method in
        the model init method.

        Notes
        -----
        Subsequent calls to this method with an :class:`~scvi.data.AnnDataManager` instance
        referring to the same underlying AnnData object will overwrite the reference to previous
        :class:`~scvi.data.AnnDataManager`.
        """
        mdata_id = mudata_manager.adata_uuid
        cls._setup_mdata_manager_store[mdata_id] = mudata_manager

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

    @classmethod
    @abstractmethod
    def setup_mudata(
        cls,
        mdata: MuData,
        *args,
        **kwargs,
    ):
        """%(summary)s.

        Each model class deriving from this class provides parameters to this method
        according to its needs. To operate correctly with the model initialization,
        the implementation must call :meth:`~scvi.model.base.BaseModelClass.register_manager`
        on a model-specific instance of :class:`~scvi.data.AnnDataManager`.
        """
