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
        train: bool = True,
    ) -> None:
        super().__init__()

        self.mdata = mdata
        self.rna = np.array(mdata[rna_mod].layers["counts"].todense()).reshape(-1)  # type: ignore
        self.atac = np.array(mdata[atac_mod].layers["counts"].todense())  # type: ignore
        self.tf = np.array(mdata[rna_mod].obsm["tf"].todense())  # type: ignore

        # convert seq to one-hot encoding
        self.peak_seqs = self.mdata[atac_mod].var["dna_sequence"].values.tolist()
        self.peak_seqs = encode_seq(self.peak_seqs)

        self.train = train

        self.len = self.mdata.n_obs

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_map = {}
        if self.train:
            data_map["atac"] = self.atac[idx]
            data_map["rna"] = self.rna[idx]
            data_map["dna"] = self.peak_seqs
            data_map["tf"] = self.tf[idx]
        else:
            data_map["atac"] = self.atac[idx]
            data_map["dna"] = self.peak_seqs
            data_map["tf"] = self.tf[idx]

        return data_map
