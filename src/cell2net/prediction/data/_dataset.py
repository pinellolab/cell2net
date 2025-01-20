from collections.abc import Sequence

import numpy as np
from mudata import MuData
from torch.utils.data import Dataset

from ._utils import encode_seq


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
    >>> dataset = MuTorchDataset(
    ...     mdata=mdata,
    ...     rna_mod="rna",
    ...     atac_mod="atac",
    ...     covariates=["age", "sex"],
    ...     train=True
    ... )
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
        self.target_exp = np.array(mdata[rna_mod].layers["counts"].todense(), dtype=np.float32).reshape(-1)  # type: ignore
        self.peak_acc = np.array(mdata[atac_mod].layers["counts"].todense(), dtype=np.float32)  # type: ignore
        self.tf_exp = np.array(mdata[rna_mod].obsm["tf"].todense(), dtype=np.float32)  # type: ignore

        self.covariates = mdata.obs[covariates].to_numpy(dtype=np.float32)

        # distance of peak to TSS, normalized by the maximum value
        self.peak_dist = np.array(
            mdata.uns["peak_to_gene"]["distance"].values, dtype=np.float32
        )
        self.peak_dist = np.exp(-self.peak_dist / 500000).astype(np.float32)

        # convert seq to one-hot encoding
        self.peak_seq = encode_seq(
            self.mdata[atac_mod].var["dna_sequence"].values.tolist()
        )

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
