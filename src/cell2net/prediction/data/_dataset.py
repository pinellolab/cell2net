from collections.abc import Sequence

import numpy as np
import torch
from scipy.sparse import issparse
from typing import Optional
from mudata import MuData
import pandas as pd
from torch.utils.data import Dataset

from cell2net.preprocessing import seq_to_one_hot
from cell2net._logging import logger


def to_dense(x, dtype=np.float32) -> np.ndarray:
    """
    Convert an AnnData layer/obsm entry to a dense C-contiguous ndarray.

    Handles scipy sparse, np.matrix, plain ndarray, pandas DataFrame,
    dask arrays, and anndata's backed sparse datasets.
    """
    # Backed sparse (anndata CSRDataset/CSCDataset): materialize first
    if hasattr(x, "to_memory"):
        x = x.to_memory()

    # Dask / other lazy arrays
    if hasattr(x, "compute"):
        x = x.compute()

    if issparse(x):
        x = x.toarray()          # prefer toarray() over todense(): returns ndarray, not np.matrix
    elif hasattr(x, "to_numpy"):  # pandas DataFrame / Series
        x = x.to_numpy()
    elif hasattr(x, "toarray"):   # sparse-like not caught by issparse (e.g. sparse arrays)
        x = x.toarray()
    else:
        x = np.asarray(x)        # also unwraps np.matrix into a base ndarray

    return np.ascontiguousarray(x, dtype=dtype)


class MuTorchDataset(Dataset):
    """
    A PyTorch Dataset for single-cell multi-modal data.

    This dataset is designed for multi-modal data stored in a MuData object,
    including RNA expression, ATAC accessibility, transcription factor (TF) activity,
    covariates, and peak-to-gene associations.

    Parameters
    ----------
    mdata:
        A MuData object containing multi-modal data.
        - RNA data should be stored in the `rna_mod` modality, with count data in `layers["counts"]` and transcription factor (TF) activity in `obsm["tf"]`.
        - ATAC data should be stored in the `atac_mod` modality, with count data in `layers["counts"]` and peak sequences in `var["dna_sequence"]`.
        - Peak-to-gene associations should be in `uns["peak_to_gene"]`, with a column "distance" specifying distances from peaks to transcription start sites (TSS).
    rna_mod:
        The modality name for RNA data in the MuData object, by default "rna".
    atac_mod:
        The modality name for ATAC data in the MuData object, by default "atac".
    covariates: Sequence[str], optional
        A list of column names in `mdata.obs` representing covariates to include in the dataset.
        If None, no covariates are included, by default None.
    train:
        Whether the dataset is used for training.
        If True, the `target_exp` (target expression) will be included in the output, by default True.

    Attributes
    ----------
    mdata : MuData
        The MuData object used to create the dataset.

    target_exp : np.ndarray
        RNA expression data from the `rna_mod` modality, flattened to a 1D array.

    peak_acc : np.ndarray
        ATAC accessibility data from the `atac_mod` modality.

    tf_exp : np.ndarray
        Transcription factor activity data from the `rna_mod` modality.

    covariates : np.ndarray
        Covariate data extracted from `mdata.obs`.

    peak_dist : np.ndarray
        Peak-to-TSS distances, transformed using an exponential decay function and normalized.

    peak_seq : np.ndarray
        One-hot encoded sequences of peaks from the `atac_mod` modality.

    train : bool
        Indicates whether the dataset is for training or not.

    len : int
        The number of observations in the dataset.

    Example
    -------
    >>> dataset = MuTorchDataset(mdata=mdata, rna_mod="rna", atac_mod="atac", covariates=["age", "sex"], train=True)
    >>> print(len(dataset))
    10000
    >>> data = dataset[0]
    >>> print(data["peak_acc"].shape)
    (500,)
    >>> print(data["peak_seq"].shape)
    (500, 4)
    >>> print(data["target_exp"].shape)
    (1,)
    """

    def __init__(
        self,
        mdata: MuData,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        covariates: Sequence[str] | None = None,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.mdata = mdata

        if mdata is None:
            raise ValueError("mdata is required")

        self.target_exp = to_dense(mdata[rna_mod].layers["counts"]).ravel() # type: ignore
        self.peak_acc = to_dense(mdata[atac_mod].layers["counts"]) # type: ignore
        self.tf_exp = to_dense(mdata[rna_mod].obsm["tf"]) # type: ignore

        # self.target_exp = np.array(mdata[rna_mod].layers["counts"].todense(), dtype=np.float32).reshape(-1)  # type: ignore
        # self.peak_acc = np.array(mdata[atac_mod].layers["counts"].todense(), dtype=np.float32)  # type: ignore
        # self.tf_exp = np.array(mdata[rna_mod].obsm["tf"].todense(), dtype=np.float32)  # type: ignore

        self.covariates = mdata.obs[covariates].to_numpy(dtype=np.float32)

        # distance of peak to TSS, normalized by the maximum value
        self.peak_dist = np.array(mdata.uns["peak_to_gene"]["distance"].values, dtype=np.float32)
        self.peak_dist = np.exp(-self.peak_dist / 500000).astype(np.float32)

        # convert sequence to one-hot encoding
        self.peak_seq = []
        for seq in self.mdata[atac_mod].uns["peaks"]["sequence"].values.tolist():
            # Ensure seq_to_one_hot is defined or imported
            one_hot_encode = seq_to_one_hot(seq)
            if one_hot_encode is None:
                logger.error(f"Failed to encode sequence: {seq}")

            self.peak_seq.append(torch.from_numpy(one_hot_encode))

        self.peak_seq = torch.stack(self.peak_seq)

        self.train = train
        self.len = self.mdata.n_obs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_map = {}
        data_map["peak_acc"] = self.peak_acc[idx]
        data_map["peak_seq"] = self.peak_seq
        data_map["peak_dist"] = self.peak_dist
        data_map["tf_exp"] = self.tf_exp[idx]
        data_map["covariates"] = self.covariates[idx]

        if self.train:
            data_map["target_exp"] = self.target_exp[idx]

        return data_map

class MuTorchDatasetWithGenotype(Dataset):
    def __init__(
        self,
        mdata: Optional[MuData] = None,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        peak_to_gene_key: str = "peak_to_gene",
        peak_seq_key: str = "personal_genome_seq",
        covariates: Sequence[str] | None = None,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.rna_mod = rna_mod
        self.atac_mod = atac_mod

        if mdata is None:
            raise ValueError("mdata is required")

        self.target_exp = to_dense(mdata[rna_mod].layers["counts"]).ravel() # type: ignore
        self.peak_acc = to_dense(mdata[atac_mod].layers["counts"]) # type: ignore
        self.tf_exp = to_dense(mdata[rna_mod].obsm["tf"]) # type: ignore

        # self.target_exp = np.array(mdata[rna_mod].layers["counts"].todense(), dtype=np.float32).reshape(-1)  # type: ignore
        # self.peak_acc = np.array(mdata[atac_mod].layers["counts"].todense(), dtype=np.float32)  # type: ignore
        # self.tf_exp = np.array(mdata[rna_mod].obsm["tf"].todense(), dtype=np.float32)  # type: ignore
        self.covariates = mdata.obs[covariates].to_numpy(dtype=np.float32)

        # distance of peak to TSS, normalized by the maximum value
        self.peak_dist = np.array(mdata.uns[peak_to_gene_key]["distance"].values, dtype=np.float32)
        self.peak_dist = np.exp(-self.peak_dist / 500000).astype(np.float32)

        # prepare peak sequence
        # subset seq to only include peaks in peak_to_gene
        self.donors = mdata.obs["donor"].values.tolist()

        self.df_seq = mdata[atac_mod].uns[peak_seq_key]
        self.df_seq = self.df_seq[
            self.df_seq["peak"].isin(mdata.uns[peak_to_gene_key]["peak"].values.tolist())
        ].reset_index(drop=True)

        # convert DNA sequence to one-hot encoding for each donor
        self.peak_seq1 = {}
        self.peak_seq2 = {}
        for donor in list(set(self.donors)):
            _df_seq = self.df_seq[self.df_seq["donor"] == donor].set_index("peak")

            # resort the seq to the same order as peak_acc
            _df_seq = _df_seq.loc[mdata[atac_mod].var_names.tolist()]

            # encode seq
            seq_1, seq_2 = [], []
            for seq in _df_seq["seq_1"].values.tolist():
                one_hot_encode = seq_to_one_hot(seq)
                if one_hot_encode is None:
                    logger.error(f"Failed to encode sequence: {seq}")

                seq_1.append(torch.from_numpy(one_hot_encode))

            for seq in _df_seq["seq_2"].values.tolist():
                one_hot_encode = seq_to_one_hot(seq)
                if one_hot_encode is None:
                    logger.error(f"Failed to encode sequence: {seq}")

                seq_2.append(torch.from_numpy(one_hot_encode))

            seq_1 = torch.stack(seq_1)
            seq_2 = torch.stack(seq_2)
            self.peak_seq1[donor] = seq_1
            self.peak_seq2[donor] = seq_2

        self.train = train
        self.len = mdata.n_obs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_map = {}
        data_map["peak_seq1"] = self.peak_seq1[self.donors[idx]]
        data_map["peak_seq2"] = self.peak_seq2[self.donors[idx]]
        data_map["peak_acc"] = self.peak_acc[idx]
        data_map["peak_dist"] = self.peak_dist
        data_map["tf_exp"] = self.tf_exp[idx]
        data_map["covariates"] = self.covariates[idx]

        if self.train:
            data_map["target_exp"] = self.target_exp[idx]

        return data_map

class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
    ) -> None:
        super().__init__()
        self.df = df
        # convert sequence to one-hot encoding
        self.peak_seq = []
        for seq in self.df['sequence'].values.tolist():
            one_hot_encode = seq_to_one_hot(seq)
            if one_hot_encode is None:
                logger.error(f"Failed to encode sequence: {seq}")

            self.peak_seq.append(torch.from_numpy(one_hot_encode))

        self.peak_seq = torch.stack(self.peak_seq)

        # add 1 dim for channel
        self.peak_seq = self.peak_seq.unsqueeze(1)  # (n_peaks, 1, peak_len, 4)
        self.peak_acc = df['acc'].values.tolist()
        self.len = self.df.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_map = {}
        data_map["peak_acc"] = self.peak_acc[idx] # all cells for peak idx
        data_map["peak_seq"] = self.peak_seq[idx] # sequence for peak idx

        return data_map
