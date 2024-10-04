from __future__ import annotations

from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import torch
from mudata import MuData
from scipy.sparse import issparse
from torch.utils.data import DataLoader, Dataset

# avoid circular imports for type annotation.
if TYPE_CHECKING:
    from ._manager import MuDataManager

from ._utils import encode_seq, registry_key_to_default_dtype


class MuTorchDataset(Dataset):
    def __init__(
        self,
        mdata_manager: MuDataManager,
        getitem_tensors: list | dict[str, type] | None = None,
        load_sparse_tensor: bool = False,
    ) -> None:
        super().__init__()

        if mdata_manager.mdata is None:
            raise ValueError(
                "Please run ``register_fields`` on ``mdata_manager`` first."
            )
        self.mdata_manager = mdata_manager
        self.keys_and_dtypes = getitem_tensors
        self.load_sparse_tensor = load_sparse_tensor

        # self.atac = atac
        # self.rna = rna
        # self.train = train
        # self.len = atac.shape[0]

        # # convert seq to one-hot encoding
        # self.peak_seq = encode_seq(seq_list)

    @property
    def registered_keys(self):
        """Keys in the data registry."""
        return self.mdata_manager.data_registry.keys()

    @property
    def keys_and_dtypes(self):
        """Keys and corresponding :class:`~np.dtype` of data to fetch in ``__getitem__``."""
        return self._keys_and_dtypes

    @keys_and_dtypes.setter
    def keys_and_dtypes(self, getitem_tensors: list | dict[str, type] | None):
        """Set keys and corresponding :class:`~np.dtype` of data to fetch in ``__getitem__``.

        Raises an error if any of the keys are not in the data registry.
        """
        if isinstance(getitem_tensors, list):
            keys_to_dtypes = {
                key: registry_key_to_default_dtype(key) for key in getitem_tensors
            }
        elif isinstance(getitem_tensors, dict):
            keys_to_dtypes = getitem_tensors
        elif getitem_tensors is None:
            keys_to_dtypes = {
                key: registry_key_to_default_dtype(key) for key in self.registered_keys
            }
        else:
            raise ValueError("`getitem_tensors` must be a `list`, `dict`, or `None`")

        for key in keys_to_dtypes:
            if key not in self.registered_keys:
                raise KeyError(f"{key} not found in the data registry.")

        self._keys_and_dtypes = keys_to_dtypes

    @property
    def data(self):
        """Dictionary of data tensors.

        First time this is accessed, data is fetched from the underlying
        :class:`~mudata.MuData` object. Subsequent accesses will return the
        cached dictionary.
        """
        if not hasattr(self, "_data"):
            self._data = {
                key: self.mdata_manager.get_from_registry(key)
                for key in self.keys_and_dtypes
            }
        return self._data

    def __len__(self):
        return self.mdata_manager.mdata.shape[0]

    def __getitem__(
        self, indexes: int | list[int] | slice
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Fetch data from the :class:`~anndata.AnnData` object.

        Parameters
        ----------
        indexes
            Indexes of the observations to fetch. Can be a single index, a list of indexes, or a
            slice.

        Returns
        -------
        Mapping of data registry keys to arrays of shape ``(n_obs, ...)``.
        """
        if isinstance(indexes, int):
            indexes = [indexes]  # force batched single observations

        if self.mdata_manager.mdata.isbacked and isinstance(indexes, list | np.ndarray):
            # need to sort indexes for h5py datasets
            indexes = np.sort(indexes)

        data_map = {}

        for key, dtype in self.keys_and_dtypes.items():
            data = self.data[key]

            if isinstance(data, np.ndarray | h5py.Dataset):
                sliced_data = data[indexes].astype(dtype, copy=False)
            elif isinstance(data, pd.DataFrame):
                sliced_data = data.iloc[indexes, :].to_numpy().astype(dtype, copy=False)
            elif issparse(data) or isinstance(data, SparseDataset):
                sliced_data = data[indexes].astype(dtype, copy=False)
                if self.load_sparse_tensor:
                    sliced_data = scipy_to_torch_sparse(sliced_data)
                else:
                    sliced_data = sliced_data.toarray()
            elif isinstance(data, str) and key == REGISTRY_KEYS.MINIFY_TYPE_KEY:
                # for minified  anndata, we need this because we can have a string
                # for `data``, which is the value of the MINIFY_TYPE_KEY in adata.uns,
                # used to record the type data minification
                # TODO: Adata manager should have a list of which fields it will load
                continue
            else:
                raise TypeError(f"{key} is not a supported type")

            data_map[key] = sliced_data

        return data_map


class MuTorchDatasetSimple(Dataset):
    def __init__(
        self,
        mdata: MuData,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.mdata = mdata
        self.adata_rna = mdata["rna"]
        self.adata_atac = mdata["atac"]

        self.train = train
        self.len = self.mdata.n_obs
        self.peak_seqs = self.mdata["atac"].var["dna_sequence"].values.tolist()

        # convert seq to one-hot encoding
        self.peak_seqs = encode_seq(self.peak_seqs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = {}
        if self.train:
            data["atac"] = np.array(
                self.adata_atac[idx].layers["counts"].todense()
            ).reshape(-1)
            data["rna"] = np.array(
                self.adata_rna[idx].layers["counts"].todense()
            ).reshape(-1)
            data["dna"] = self.peak_seqs
        else:
            data["atac"] = np.array(
                self.adata_atac[idx].layers["counts"].todense()
            ).reshape(-1)
            data["dna"] = self.peak_seqs

        return data


def get_dataloader(
    mdata: MuData,
    batch_size: int = 128,
    num_workers: int = 8,
    drop_last: bool = False,
    shuffle: bool = True,
    train: bool = True,
):
    dataset = MuTorchDatasetSimple(
        mdata=mdata,
        train=train,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=drop_last,
        persistent_workers=True,
    )

    return dataloader
