import numpy as np
from mudata import MuData
from torch.utils.data import Dataset

from ._utils import encode_seq


class MuTorchDataset(Dataset):
    def __init__(
        self,
        mdata: MuData,
        rna_mod: str = "rna",
        atac_mod: str = "atac",
        covariates: list[str] | None = None,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.mdata = mdata
        self.target_exp = np.array(mdata[rna_mod].layers["counts"].todense()).reshape(-1)  # type: ignore
        self.peak_acc = np.array(mdata[atac_mod].layers["counts"].todense())  # type: ignore
        self.tf_exp = np.array(mdata[rna_mod].obsm["tf"].todense())  # type: ignore

        self.covariates = mdata.obs[covariates].to_numpy()

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
        data_map["tf_exp"] = self.tf_exp[idx]
        data_map["covariates"] = self.covariates[idx]

        if self.train:
            data_map["target_exp"] = self.target_exp[idx]

        return data_map
