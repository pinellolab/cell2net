import torch
from torch.utils.data import DataLoader, Dataset

from ._utils import one_hot_encode


def _encode_seq(seq_list):
    data = []
    for seq in seq_list:
        data.append(one_hot_encode(seq))

    data = torch.stack(data)

    return data


class MultiOmeDataSet(Dataset):
    def __init__(
        self,
        seq_list: list,
        atac: torch.Tensor,
        rna: torch.Tensor,
        train: bool = True,
    ) -> None:
        super().__init__()

        self.atac = atac
        self.rna = rna
        self.train = train
        self.len = atac.shape[0]

        # convert seq to one-hot encoding
        self.peak_seq = _encode_seq(seq_list)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.train:
            return (self.peak_seq, self.atac[idx], self.rna[idx])
        else:
            return (self.peak_seq, self.atac[idx])


def get_dataloader(
    seq_list: list,
    atac: torch.Tensor,
    rna: torch.Tensor,
    batch_size: int = 128,
    num_workers: int = 8,
    drop_last: bool = False,
    shuffle: bool = True,
    train: bool = True,
):
    dataset = MultiOmeDataSet(
        seq_list=seq_list,
        atac=atac,
        rna=rna,
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


# if __name__ == "__main__":
#     peaks = pd.read_csv("")

#     ds = MultiOmeDataSet()
