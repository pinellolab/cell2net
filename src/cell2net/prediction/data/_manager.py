from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from mudata import MuData
from scvi.utils import attrdict
from torch.utils.data import Subset

import cell2net

from . import _constants
from ._base import MuDataField
from ._dataset import MuTorchDataset
from ._utils import _assign_mdata_uuid, _check_mudata_fully_paired


@dataclass
class MuDataManagerValidationCheck:
    """Validation checks for MuData compat.

    Parameters
    ----------
    check_if_view
        If True, checks if MuData is a view.
    check_fully_paired_mudata
        If True, checks if MuData is fully paired across mods.
    """

    check_if_view: bool = True
    check_fully_paired_mudata: bool = True


class MuDataManager:
    """Provides an interface to validate and process an MuData object for use in cell2net.

    A class which wraps a collection of MuDataField instances and provides an interface
    to validate and process an MuDataField object with respect to the fields.

    This is based on scvi.data.AnnDataManager in scvi-tools.

    Parameters
    ----------
    fields
        List of MuDataFields to intialize with.
    setup_method_args
        Dictionary describing the model and arguments passed in by the user
        to setup this MuDataManager.
    validation_checks
        DataClass specifying which global validation checks to run on the data object.

    Examples
    --------
    >>> fields = [LayerField("counts", "raw_counts")]
    >>> mdata_manager = MuDataManager(fields=fields)
    >>> mdata_manager.register_fields(mdata)

    Notes
    -----
    This class is not initialized with a specific AnnData object, but later sets ``self.adata``
    via :meth:`~scvi.data.AnnDataManager.register_fields`. This decouples the generalized
    definition of the scvi-tools interface with the registration of an instance of data.

    See further usage examples in the following tutorials:

    1. :doc:`/tutorials/notebooks/dev/data_tutorial`
    """

    def __init__(
        self,
        fields: list[MuDataField] | None = None,
        setup_method_args: dict | None = None,
        validation_checks: MuDataManagerValidationCheck | None = None,
    ) -> None:
        self.id = str(uuid4())
        self.mdata = None
        self.fields = fields or []
        self.validation_checks = validation_checks or MuDataManagerValidationCheck()
        self._registry = {
            _constants._CELL2NET_VERSION_KEY: cell2net.__version__,
            _constants._MODEL_NAME_KEY: None,
            _constants._SETUP_ARGS_KEY: None,
            _constants._FIELD_REGISTRIES_KEY: defaultdict(dict),
        }
        if setup_method_args is not None:
            self._registry.update(setup_method_args)

    def _assert_mudata_registered(self):
        """Asserts that an MuData object has been registered with this instance."""
        if self.mdata is None:
            raise AssertionError("MuData object not registered. Please call register_fields.")

    def _validate_mudata_object(self, mdata: MuData):
        """For a given MuData object, runs general compatibility checks."""
        # if self.validation_checks.check_if_view:
        #     _check_if_view(mdata, copy_if_view=False)

        if self.validation_checks.check_fully_paired_mudata:
            _check_mudata_fully_paired(mdata)

    def _get_setup_method_args(self) -> dict:
        """Returns the ``setup_anndata`` method arguments.

        Returns the ``setup_anndata`` method arguments, including the model name,
        that were used to initialize this :class:`~scvi.data.AnnDataManager` instance
        in the form of a dictionary.
        """
        return {
            k: v for k, v in self._registry.items() if k in {_constants._MODEL_NAME_KEY, _constants._SETUP_ARGS_KEY}
        }

    def _assign_uuid(self):
        """Assigns a UUID unique to the AnnData object.

        If already present, the UUID is left alone.
        """
        self._assert_mudata_registered()

        _assign_mdata_uuid(self.mdata)

        scvi_uuid = self.mdata.uns[_constants._CELL2NET_UUID_KEY]
        self._registry[_constants._CELL2NET_UUID_KEY] = scvi_uuid

    def _assign_most_recent_manager_uuid(self):
        """Assigns a last manager UUID to the AnnData object for future validation."""
        self._assert_mudata_registered()

        self.mdata.uns[_constants._MANAGER_UUID_KEY] = self.id

    def register_fields(
        self,
        mdata: MuData,
        source_registry: dict | None = None,
        **transfer_kwargs,
    ):
        """Registers each field associated with this instance with the AnnData object.

        Either registers or transfers the setup from `source_setup_dict` if passed in.
        Sets ``self.adata``.

        Parameters
        ----------
        mdata
            MuData object to be registered.
        source_registry
            Registry created after registering an MuData using an
            :class:`~scvi.data.AnnDataManager` object.
        transfer_kwargs
            Additional keywords which modify transfer behavior. Only applicable if
            ``source_registry`` is set.
        """
        if self.mdata is not None:
            raise AssertionError("Existing MuData object registered with this Manager instance.")

        if source_registry is None and transfer_kwargs:
            raise TypeError(
                f"register_fields() got unexpected keyword arguments {transfer_kwargs} passed "
                "without a source_registry."
            )

        self._validate_mudata_object(mdata=mdata)

        # for field in self.fields:
        #     self._add_field(
        #         field=field,
        #         mdata=mdata,
        #         source_registry=source_registry,
        #         **transfer_kwargs,
        #     )

        # Save arguments for register_fields.
        self._source_registry = deepcopy(source_registry)
        self._transfer_kwargs = deepcopy(transfer_kwargs)

        self.mdata = mdata
        self._assign_uuid()
        self._assign_most_recent_manager_uuid()

    # def _add_field(
    #     self,
    #     field: MuDataField,
    #     mdata: MuData,
    #     source_registry: dict | None = None,
    #     **transfer_kwargs,
    # ):
    #     """Internal function for adding a field with optional transferring."""
    #     field_registries = self._registry[_constants._FIELD_REGISTRIES_KEY]
    #     field_registries[field.registry_key] = {
    #         _constants._DATA_REGISTRY_KEY: field.get_data_registry(),
    #         _constants._STATE_REGISTRY_KEY: {},
    #     }
    #     field_registry = field_registries[field.registry_key]

    #     # A field can be empty if the model has optional fields (e.g. extra covariates).
    #     # If empty, we skip registering the field.
    #     if not field.is_empty:
    #         # Transfer case: Source registry is used for validation and/or setup.
    #         if source_registry is not None:
    #             field_registry[_constants._STATE_REGISTRY_KEY] = field.transfer_field(
    #                 source_registry[_constants._FIELD_REGISTRIES_KEY][
    #                     field.registry_key
    #                 ][_constants._STATE_REGISTRY_KEY],
    #                 mdata,
    #                 **transfer_kwargs,
    #             )
    #         else:
    #             field_registry[_constants._STATE_REGISTRY_KEY] = field.register_field(
    #                 mdata
    #             )
    #     # Compute and set summary stats for the given field.
    #     state_registry = field_registry[_constants._STATE_REGISTRY_KEY]
    #     field_registry[_constants._SUMMARY_STATS_KEY] = field.get_summary_stats(
    #         state_registry
    #     )

    @property
    def adata_uuid(self) -> str:
        """Returns the UUID for the MuData object registered with this instance."""
        self._assert_mudata_registered()

        return self._registry[_constants._CELL2NET_UUID_KEY]

    @property
    def registry(self) -> dict:
        """Returns the top-level registry dictionary for the MuData object."""
        return self._registry

    @property
    def data_registry(self) -> dict:
        """Returns the data registry for the MuData object registered with this instance."""
        self._assert_mudata_registered()
        return self._get_data_registry_from_registry(self._registry)

    def create_torch_dataset(
        self,
        indices: Sequence[int] | Sequence[bool],
        data_and_attributes: list[str] | dict[str, np.dtype] | None = None,
        load_sparse_tensor: bool = False,
    ) -> MuTorchDataset:
        """
        Creates a torch dataset from the MuData object registered with this instance.

        Parameters
        ----------
        indices
            The indices of the observations in the adata to use
        data_and_attributes
            Dictionary with keys representing keys in data registry
            (``adata_manager.data_registry``) and value equal to desired numpy loading type (later
            made into torch tensor) or list of such keys. A list can be used to subset to certain
            keys in the event that more tensors than needed have been registered. If ``None``,
            defaults to all registered data.
        load_sparse_tensor
            ``EXPERIMENTAL`` If ``True``, loads data with sparse CSR or CSC layout as a
            :class:`~torch.Tensor` with the same layout. Can lead to speedups in data transfers to
            GPUs, depending on the sparsity of the data.

        Returns
        -------
        :class:`~scvi.data.AnnTorchDataset`
        """
        dataset = MuTorchDataset(
            self,
            getitem_tensors=data_and_attributes,
            load_sparse_tensor=load_sparse_tensor,
        )

        if indices is not None:
            # This is a lazy subset, it just remaps indices
            dataset = Subset(dataset, indices)

        return dataset

    @staticmethod
    def _get_data_registry_from_registry(registry: dict) -> attrdict:
        data_registry = {}
        for registry_key, field_registry in registry[_constants._FIELD_REGISTRIES_KEY].items():
            field_data_registry = field_registry[_constants._DATA_REGISTRY_KEY]
            if field_data_registry:
                data_registry[registry_key] = field_data_registry
        return attrdict(data_registry)
