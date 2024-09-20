import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from torch.utils.data import DataLoader, Dataset

from ._utils import one_hot_encode


class MultiOmeDataSet(Dataset):
    def __init__(
        self,
        peak_to_gene: pd.DataFrame,
        adata_atac: AnnData,
        adata_rna: AnnData,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.peak_to_gene = peak_to_gene
        self.adata_atac = adata_atac
        self.adata_rna = adata_rna
        self.train = train
        self.len = adata_atac.shape[0]

        # convert seq to one-hot encoding
        self.peak_seq = self.encode_peak_seq()

    def encode_peak_seq(self):
        peak_seq = []
        for seq in self.peaks["Seq"].values.tolist():
            peak_seq.append(one_hot_encode(seq))

        peak_seq = torch.stack(peak_seq)

        return peak_seq

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.train:
            return (self.peak_seq, self.atac[idx], self.rna[idx])
        else:
            return (self.peak_seq, self.atac[idx])


def get_dataloader(
    peaks: pd.DataFrame,
    atac: np.array,
    rna: np.array | None,
    batch_size: int = 128,
    num_workers: int = 8,
    drop_last: bool = False,
    shuffle: bool = True,
    train: bool = True,
):
    dataset = MultiOmeDataSet(peaks=peaks, atac=atac, rna=rna, train=train)

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


if __name__ == "__main__":
    peaks = pd.read_csv("")

    ds = MultiOmeDataSet()
